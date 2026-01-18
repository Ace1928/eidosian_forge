from __future__ import annotations
import asyncio
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import (
import yaml
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.agents.agent_types import AgentType
from langchain.agents.tools import InvalidTool
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.utilities.asyncio import asyncio_timeout
@root_validator()
def validate_tools(cls, values: Dict) -> Dict:
    """Validate that tools are compatible with agent."""
    agent = values['agent']
    tools = values['tools']
    allowed_tools = agent.get_allowed_tools()
    if allowed_tools is not None:
        if set(allowed_tools) != set([tool.name for tool in tools]):
            raise ValueError(f'Allowed tools ({allowed_tools}) different than provided tools ({[tool.name for tool in tools]})')
    return values