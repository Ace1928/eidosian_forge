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
class RunnableMultiActionAgent(BaseMultiActionAgent):
    """Agent powered by runnables."""
    runnable: Runnable[dict, Union[List[AgentAction], AgentFinish]]
    'Runnable to call to get agent actions.'
    input_keys_arg: List[str] = []
    return_keys_arg: List[str] = []
    stream_runnable: bool = True
    'Whether to stream from the runnable or not. \n    \n    If True then underlying LLM is invoked in a streaming fashion to make it possible \n        to get access to the individual LLM tokens when using stream_log with the Agent \n        Executor. If False then LLM is invoked in a non-streaming fashion and \n        individual LLM tokens will not be available in stream_log.\n    '

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return self.return_keys_arg

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        Returns:
            List of input keys.
        """
        return self.input_keys_arg

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks=None, **kwargs: Any) -> Union[List[AgentAction], AgentFinish]:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, **{'intermediate_steps': intermediate_steps}}
        final_output: Any = None
        if self.stream_runnable:
            for chunk in self.runnable.stream(inputs, config={'callbacks': callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={'callbacks': callbacks})
        return final_output

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks=None, **kwargs: Any) -> Union[List[AgentAction], AgentFinish]:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, **{'intermediate_steps': intermediate_steps}}
        final_output: Any = None
        if self.stream_runnable:
            async for chunk in self.runnable.astream(inputs, config={'callbacks': callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(inputs, config={'callbacks': callbacks})
        return final_output