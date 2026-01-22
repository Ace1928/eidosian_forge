from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
from typing_extensions import get_args
from langchain_core.language_models import LanguageModelOutput
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import run_in_executor
class BaseGenerationOutputParser(BaseLLMOutputParser, RunnableSerializable[LanguageModelOutput, T]):
    """Base class to parse the output of an LLM call."""

    @property
    def InputType(self) -> Any:
        return Union[str, AnyMessage]

    @property
    def OutputType(self) -> Type[T]:
        return T

    def invoke(self, input: Union[str, BaseMessage], config: Optional[RunnableConfig]=None) -> T:
        if isinstance(input, BaseMessage):
            return self._call_with_config(lambda inner_input: self.parse_result([ChatGeneration(message=inner_input)]), input, config, run_type='parser')
        else:
            return self._call_with_config(lambda inner_input: self.parse_result([Generation(text=inner_input)]), input, config, run_type='parser')

    async def ainvoke(self, input: Union[str, BaseMessage], config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> T:
        if isinstance(input, BaseMessage):
            return await self._acall_with_config(lambda inner_input: self.aparse_result([ChatGeneration(message=inner_input)]), input, config, run_type='parser')
        else:
            return await self._acall_with_config(lambda inner_input: self.aparse_result([Generation(text=inner_input)]), input, config, run_type='parser')