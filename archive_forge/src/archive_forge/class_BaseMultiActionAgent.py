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
class BaseMultiActionAgent(BaseModel):
    """Base Multi Action Agent class."""

    @property
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return ['output']

    def get_allowed_tools(self) -> Optional[List[str]]:
        return None

    @abstractmethod
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks=None, **kwargs: Any) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Actions specifying what tool to use.
        """

    @abstractmethod
    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks=None, **kwargs: Any) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Actions specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(self, early_stopping_method: str, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == 'force':
            return AgentFinish({'output': 'Agent stopped due to max iterations.'}, '')
        else:
            raise ValueError(f'Got unsupported early_stopping_method `{early_stopping_method}`')

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        try:
            _dict['_type'] = str(self._agent_type)
        except NotImplementedError:
            pass
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Example:
        .. code-block:: python

            # If working with agent executor
            agent.agent.save(file_path="path/agent.yaml")
        """
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path
        agent_dict = self.dict()
        if '_type' not in agent_dict:
            raise NotImplementedError(f'Agent {self} does not support saving.')
        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)
        if save_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith(('.yaml', '.yml')):
            with open(file_path, 'w') as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            raise ValueError(f'{save_path} must be json or yaml')

    def tool_run_logging_kwargs(self) -> Dict:
        return {}