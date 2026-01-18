    test_output = code_qwen.run(chat_template, stop_sequences=["10/10"], stream=False)     
                  ^^^^^^^^^^^^^
AttributeError: 'TransformersModel' object has no attribute 'run'
Traceback (most recent call last):
  File "c:\minilm\tool_web.py", line 234, in <module>
    memory_manager_agent = CodeAgent(
                           ^^^^^^^^^^
  File "C:\Users\ace19\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\smolagents\agents.py", line 838, in __init__
    super().__init__(
  File "C:\Users\ace19\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\smolagents\agents.py", line 195, in __init__
    self.managed_agents = {agent.name: agent for agent in managed_agents}
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ace19\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\smolagents\agents.py", line 195, in <dictcomp>
    self.managed_agents = {agent.name: agent for agent in managed_agents}
                           ^^^^^^^^^^
AttributeError: 'ToolCallingAgent' object has no attribute 'name'


You need to fix this. Making minimal changes. The tool agent shouldnt have a name.
The code_qwen doesnt have a run method, it is the specific instances of the agents that have run methods etc. 

come on....

HuggingFace smolagents: The best Multi-Agent framework so far?
Comparing Autogen, Langraph, CrewAI, Magentic-One, etc
Mehul Gupta
Data Science in your pocket

Mehul Gupta
·

Published in

Data Science in your pocket
·
5 min read
·
3 days ago

Photo by Joel Rivera-Camacho on Unsplash

As you must have read in multiple places, 2025 is the year of AI-Agents. So much so that Mark Zuckerberg has openly said Meta will have Mid-Senior Engineering being AI Agents.

This is very clear from the past few releases where Microsoft now owns 3 Multi-Agent Orchestration frameworks (AutoGen, Magentic-One, Tiny-Troupe), OpenAI releases Swarm, and AWS launched multi-agent-orchestrator apart from standalone Langraph and CrewAI. Now even HuggingFace has joined the party launching “smolagents” which is yet another Multi-Agent Framework but with a difference.
What are smolagents?

Smolagents is a newly launched agent framework by Hugging Face, designed to simplify the creation of intelligent agents that leverage large language models (LLMs). This lightweight library enables developers to build agents with minimal code, focusing on practicality and ease of use.
Subscribe to datasciencepocket on Gumroad
On a Mission to teach AI to everyone !

datasciencepocket.gumroad.com

Key Features of Smolagents

    Simplicity: Smolagents allows for rapid prototyping and deployment with a straightforward coding approach, making it accessible even for those with limited experience in creating agents.

    Code-Centric Agents: The framework supports agents that execute actions directly as Python code. This method often results in higher accuracy and efficiency compared to traditional tool-based agents, which may require more steps.

    LLM Compatibility: Smolagents is designed to be LLM-agnostic, meaning it can integrate seamlessly with any LLM available on the Hugging Face Hub, as well as other popular models through its LiteLLM integration.

    Agentic Capabilities: The framework allows LLMs to control workflows by writing actions that can be executed via external tools. This flexibility enhances the ability of agents to tackle complex tasks that do not fit into predefined workflows.

    Security Features: Smolagents includes mechanisms for executing code in secure, sandboxed environments, ensuring safe operation when running potentially risky code.

More on Code-Centric Agents

This is what makes smolagents unique from all other frameworks. But let’s first understand how code agents work in other frameworks
Code Agents in other Agentic frameworks:

In most other frameworks, agents perform tasks by calling tools using JSON-like formats. Here’s how it typically works:

    Action Representation:
    The agent writes actions as structured JSON. Each action includes:

    Tool name: What tool to use (e.g., “search”, “translate”).

    Arguments: Parameters required by the tool (e.g., “query”: “current weather”).

Execution Process:

    The framework reads this JSON, figures out which tool to use, and runs the tool with the given arguments.

    Once the tool returns a result, the agent processes it and decides the next action.

How Smolagents Are Different

Instead of using JSON, Smolagents lets the agent write actual Python code to perform actions. Here’s why this is different and better:

    Action as Code:
    In Smolagents, the agent directly generates Python code to execute actions.

    Direct Execution:
    The framework runs this code directly, without needing to translate JSON into tool calls. This makes execution faster, simpler, and more accurate.

Example:

Assume you pass this command to the agentic framework

    Generate a sunset image and display

In other frameworks, this will be a more complex pipeline as two actions are to be taken separately
Other frameworks (usually)

Step 1: Generate Image

The agent sends a JSON request to generate the image:

{
  "tool": "generate_image",
  "args": { "prompt": "sunset over mountains" }
}

Step 2: Display Image

After the image is generated, the agent sends another action in JSON to display it:

{
  "tool": "display_image",
  "args": { "image": "generated_image.png" }
}

Each step involves parsing the JSON, figuring out the tool to call, and executing it. The agent cannot easily chain these actions together or reuse the result.
While using smolagents,

It will simply write a code to execute everything together

image = generate_image("sunset over mountains")
display(image)

Hence no multiple calls and less complexity
Why Smolagents is Better

    Less Overhead:
    No need to convert JSON into tool calls — just run the code.

    Greater Flexibility:
    Python code can handle complex tasks (loops, conditions, custom functions) that JSON cannot.

    Better Alignment with LLM Training:
    Since LLMs are heavily trained on Python code, they perform better when generating actions in code rather than JSON.

    Simplified Execution:
    The agent can handle tasks directly in one go, without needing extra steps to interpret JSON.

    Compatible with any local LLMs and API as well

smolagents cons

Though the framework is great, there are some perils one must know before going for it

    Buggy code execution : Even though Smolagents includes safeguards like a secure local interpreter and E2B remote execution, still, there can be cases where the framework runs something like “delete everything” code snippet making your PC faulty.

    Doesn’t look as flexible as LangGraph.

    Your LLM should be good at coding !! I tried it with a few non-coder LLMs and the results weren’t great

    Is writing code for everything necesaary? Not at all.Smolagents may introduce unnecessary complexity.

    So, is it the best framework? I dont think so but a pretty good one if you’re starting with Agentic frameworks. Plus its easy !!

How to use smolagents?

Taking an example straight from the repo

    pip install the package

pip install smolagents

2. Create your CodeAgent and pass it some tools

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

As you can see, CodeAgent is the standalone agent that would be using any sort of tools (internet search in this case). For the given prompt, CodeAgent will write codes to:

    Search the internet to figure out different information required for answering the question

    Write the code in such a way that the final output is “seconds” required, hence a completely customized code for any of your problem !!

Concluding,

The framework looks fun to use and is definitely worth giving a try. Below is the git repo to access it


Official Documentation:
Orchestrate a multi-agent system 🤖🤝🤖

In this notebook we will make a multi-agent web browser: an agentic system with several agents collaborating to solve problems using the web!

It will be a simple hierarchy, using a ManagedAgent object to wrap the managed web search agent:

              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
  Code interpreter   +--------------------------------+
       tool          |         Managed agent          |
                     |      +------------------+      |
                     |      | Web Search agent |      |
                     |      +------------------+      |
                     |         |            |         |
                     |  Web Search tool     |         |
                     |             Visit webpage tool |
                     +--------------------------------+

Let’s set up this system.

Run the line below to install the required dependencies:

!pip install markdownify duckduckgo-search smolagents --upgrade -q

Let’s login in order to call the HF Inference API:

from huggingface_hub import login

login()

⚡️ Our agent will be powered by Qwen/Qwen2.5-Coder-32B-Instruct using HfApiModel class that uses HF’s Inference API: the Inference API allows to quickly and easily run any OS model.

Note: The Inference API hosts models based on various criteria, and deployed models may be updated or replaced without prior notice. Learn more about it here.

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

🔍 Create a web search tool

For web browsing, we can already use our pre-existing DuckDuckGoSearchTool tool to provide a Google search equivalent.

But then we will also need to be able to peak into the page found by the DuckDuckGoSearchTool. To do so, we could import the library’s built-in VisitWebpageTool, but we will build it again to see how it’s done.

So let’s create our VisitWebpageTool tool from scratch using markdownify.

import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool



def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

Ok, now let’s initialize and test our tool!

print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])

Build our multi-agent system 🤖🤝🤖

Now that we have all the tools search and visit_webpage, we can use them to create the web agent.

Which configuration to choose for this agent?

    Web browsing is a single-timeline task that does not require parallel tool calls, so JSON tool calling works well for that. We thus choose a JsonAgent.
    Also, since sometimes web search requires exploring many pages before finding the correct answer, we prefer to increase the number of max_steps to 10.

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
)

model = HfApiModel(model_id)

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
)

We then wrap this agent into a ManagedAgent that will make it callable by its manager agent.

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

Finally we create a manager agent, and upon initialization we pass our managed agent to it in its managed_agents argument.

Since this agent is the one tasked with the planning and thinking, advanced reasoning will be beneficial, so a CodeAgent will be the best choice.

Also, we want to ask a question that involves the current year and does additional data calculations: so let us add additional_authorized_imports=["time", "numpy", "pandas"], just in case the agent needs these packages.

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

That’s all! Now let’s run our system! We select a question that requires both some calculation and research:

answer = manager_agent.run("If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")

We get this report as the answer:

Based on current growth projections and energy consumption estimates, if LLM trainings continue to scale up at the 
current rhythm until 2030:

