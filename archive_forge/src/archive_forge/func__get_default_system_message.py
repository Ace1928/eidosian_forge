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
def _get_default_system_message() -> SystemMessage:
    return SystemMessage(content='Do your best to answer the questions. Feel free to use any tools available to look up relevant information, only if necessary')