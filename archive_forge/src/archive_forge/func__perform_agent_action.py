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
def _perform_agent_action(self, name_to_tool_map: Dict[str, BaseTool], color_mapping: Dict[str, str], agent_action: AgentAction, run_manager: Optional[CallbackManagerForChainRun]=None) -> AgentStep:
    if run_manager:
        run_manager.on_agent_action(agent_action, color='green')
    if agent_action.tool in name_to_tool_map:
        tool = name_to_tool_map[agent_action.tool]
        return_direct = tool.return_direct
        color = color_mapping[agent_action.tool]
        tool_run_kwargs = self.agent.tool_run_logging_kwargs()
        if return_direct:
            tool_run_kwargs['llm_prefix'] = ''
        observation = tool.run(agent_action.tool_input, verbose=self.verbose, color=color, callbacks=run_manager.get_child() if run_manager else None, **tool_run_kwargs)
    else:
        tool_run_kwargs = self.agent.tool_run_logging_kwargs()
        observation = InvalidTool().run({'requested_tool_name': agent_action.tool, 'available_tool_names': list(name_to_tool_map.keys())}, verbose=self.verbose, color=None, callbacks=run_manager.get_child() if run_manager else None, **tool_run_kwargs)
    return AgentStep(action=agent_action, observation=observation)