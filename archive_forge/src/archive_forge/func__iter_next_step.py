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
def _iter_next_step(self, name_to_tool_map: Dict[str, BaseTool], color_mapping: Dict[str, str], inputs: Dict[str, str], intermediate_steps: List[Tuple[AgentAction, str]], run_manager: Optional[CallbackManagerForChainRun]=None) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
    """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
    try:
        intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)
        output = self.agent.plan(intermediate_steps, callbacks=run_manager.get_child() if run_manager else None, **inputs)
    except OutputParserException as e:
        if isinstance(self.handle_parsing_errors, bool):
            raise_error = not self.handle_parsing_errors
        else:
            raise_error = False
        if raise_error:
            raise ValueError(f'An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: {str(e)}')
        text = str(e)
        if isinstance(self.handle_parsing_errors, bool):
            if e.send_to_llm:
                observation = str(e.observation)
                text = str(e.llm_output)
            else:
                observation = 'Invalid or incomplete response'
        elif isinstance(self.handle_parsing_errors, str):
            observation = self.handle_parsing_errors
        elif callable(self.handle_parsing_errors):
            observation = self.handle_parsing_errors(e)
        else:
            raise ValueError('Got unexpected type of `handle_parsing_errors`')
        output = AgentAction('_Exception', observation, text)
        if run_manager:
            run_manager.on_agent_action(output, color='green')
        tool_run_kwargs = self.agent.tool_run_logging_kwargs()
        observation = ExceptionTool().run(output.tool_input, verbose=self.verbose, color=None, callbacks=run_manager.get_child() if run_manager else None, **tool_run_kwargs)
        yield AgentStep(action=output, observation=observation)
        return
    if isinstance(output, AgentFinish):
        yield output
        return
    actions: List[AgentAction]
    if isinstance(output, AgentAction):
        actions = [output]
    else:
        actions = output
    for agent_action in actions:
        yield agent_action
    for agent_action in actions:
        yield self._perform_agent_action(name_to_tool_map, color_mapping, agent_action, run_manager)