# Core orchestrators for agentic workflows.
from smolagents import (
    CodeAgent,  # Agent capable of executing code.
    ToolCallingAgent,  # Agent capable of calling external tools.
    TransformersModel,  # Abstraction for transformer-based language models.
    ManagedAgent,  # Agent managed by a higher-level agent.
    DuckDuckGoSearchTool,  # Tool for performing web searches using DuckDuckGo.
    USER_PROMPT_PLAN_UPDATE,
    CODE_SYSTEM_PROMPT,
    MANAGED_AGENT_PROMPT,
    TOOL_CALLING_SYSTEM_PROMPT,
    SINGLE_STEP_CODE_SYSTEM_PROMPT,
    PLAN_UPDATE_FINAL_PLAN_REDACTION
)
# Library for working with transformer models.
from transformers import AutoModelForCausalLM
# Library for optimizing large model training, including disk offloading.
from accelerate import disk_offload
# Custom utility functions for agent interactions, specifically for visiting webpages.
from eidos_tools_functions import visit_webpage

# Identifier for the language model to be used.
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# Instantiate the transformer model using the specified model ID.
model = TransformersModel(model_id)

# Define an agent capable of calling tools, specifically for web-related tasks.
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],  # Assign the web search and webpage visiting tools to this agent.
    model=model,  # Associate the language model with this agent.
    max_steps=10,  # Limit the maximum number of steps this agent can take in a single run.
)

# Wrap the web agent in a ManagedAgent for easier integration and management by other agents.
managed_web_agent = ManagedAgent(
    agent=web_agent,  # The underlying agent being managed.
    name="search",  # A descriptive name for this managed agent.
    description="Runs web searches for you. Give it your query as an argument.",  # A clear description of the agent's capabilities.
)

# Define a higher-level agent capable of executing code and managing other agents.
manager_agent = CodeAgent(
    tools=[],  # This agent does not have any direct tools, it manages other agents.
    model=model,  # Associate the language model with this agent.
    managed_agents=[managed_web_agent],  # List of agents managed by this agent.
    additional_authorized_imports=["time", "numpy", "pandas"],  # Specify additional Python libraries this agent is authorized to use in code execution.
)

# Execute a complex task using the manager agent, leveraging its ability to manage other agents and potentially execute code.
answer = manager_agent.run(task="If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")
