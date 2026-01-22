from abc import abstractmethod
from typing import Any, Optional, Protocol, Sequence, runtime_checkable
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.llms.gradient_ai import TrainResult
@runtime_checkable
class TrainableLLM(Protocol):
    """Protocol for trainable language models."""

    @abstractmethod
    def train_unsupervised(self, inputs: Sequence[str], **kwargs: Any) -> TrainResult:
        ...

    @abstractmethod
    async def atrain_unsupervised(self, inputs: Sequence[str], **kwargs: Any) -> TrainResult:
        ...