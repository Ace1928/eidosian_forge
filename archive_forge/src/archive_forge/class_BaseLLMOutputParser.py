from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
from typing_extensions import get_args
from langchain_core.language_models import LanguageModelOutput
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import run_in_executor
class BaseLLMOutputParser(Generic[T], ABC):
    """Abstract base class for parsing the outputs of a model."""

    @abstractmethod
    def parse_result(self, result: List[Generation], *, partial: bool=False) -> T:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """

    async def aparse_result(self, result: List[Generation], *, partial: bool=False) -> T:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        return await run_in_executor(None, self.parse_result, result)