from typing import Any, List, Optional  # noqa: E501
from langchain_core.language_models import BaseLanguageModel
from langchain_core.memory import BaseMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain.tools.base import BaseTool
A convenience method for creating a conversational retrieval agent.

    Args:
        llm: The language model to use, should be ChatOpenAI
        tools: A list of tools the agent has access to
        remember_intermediate_steps: Whether the agent should remember intermediate
            steps or not. Intermediate steps refer to prior action/observation
            pairs from previous questions. The benefit of remembering these is if
            there is relevant information in there, the agent can use it to answer
            follow up questions. The downside is it will take up more tokens.
        memory_key: The name of the memory key in the prompt.
        system_message: The system message to use. By default, a basic one will
            be used.
        verbose: Whether or not the final AgentExecutor should be verbose or not,
            defaults to False.
        max_token_limit: The max number of tokens to keep around in memory.
            Defaults to 2000.

    Returns:
        An agent executor initialized appropriately
    