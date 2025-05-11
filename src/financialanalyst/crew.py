from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# import litellm
# litellm._turn_on_debug()

@CrewBase
class Financialanalyst:
    """Financialanalyst crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    llm = LLM(
        # model="groq/llama-3.3-70b-versatile",
        model="gemini/gemini-2.0-flash",
        temperature=0.5,
        # seed=42,
        max_tokens=1000,
    )
    
    manager_llm = LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.5,
        # seed=42,
        max_tokens=512,
    )
    
    function_calling_llm = LLM(
        model="groq/gemma2-9b-it",
        temperature=0.5,
        # seed=42,
        max_tokens=512,
    )

    @agent
    def data_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["data_analyst_agent"],
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[
                self.scrape_tool,
                self.search_tool,
            ],
            # function_calling_llm=self.function_calling_llm,
            # max_rpm=2,
        )

    @agent
    def trading_strategy_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["trading_strategy_agent"],
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[
                self.search_tool,
                self.scrape_tool,
            ],
            # function_calling_llm=self.function_calling_llm,
            # max_rpm=2,
        )

    @agent
    def execution_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["execution_agent"],
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[
                self.search_tool,
                self.scrape_tool,
            ],
            # function_calling_llm=self.function_calling_llm,
        )

    @agent
    def risk_management_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_management_agent"],
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[
                self.search_tool,
                self.scrape_tool,
            ],
            # function_calling_llm=self.function_calling_llm,
        )

    @task
    def data_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["data_analysis_task"],
            output_file="data_analysis.md",
        )

    @task
    def trading_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["strategy_development_task"],
            output_file="trading_strategy.md",
        )

    @task
    def execution_task(self) -> Task:
        return Task(
            config=self.tasks_config["execution_planning_task"],
            output_file="execution.md",
        )
        
    @task
    def risk_management_task(self) -> Task:
        return Task(
            config=self.tasks_config["risk_assessment_task"],
            output_file="risk_management.md",
        )
    @crew
    def crew(self) -> Crew:
        """Creates the Financialanalyst crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator # type: ignore
            process=Process.hierarchical,
            verbose=True,
            manager_llm=self.manager_llm,  
        )
