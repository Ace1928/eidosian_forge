from __future__ import annotations
import asyncio
import collections
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
from contextvars import copy_context
from functools import wraps
from itertools import groupby, tee
from operator import itemgetter
from typing import (
from typing_extensions import Literal, get_args
from langchain_core._api import beta_decorator
from langchain_core.load.dump import dumpd
from langchain_core.load.serializable import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
class RunnableEachBase(RunnableSerializable[List[Input], List[Output]]):
    """Runnable that delegates calls to another Runnable
    with each element of the input sequence.

    Use only if creating a new RunnableEach subclass with different __init__ args.

    See documentation for RunnableEach for more details.
    """
    bound: Runnable[Input, Output]

    class Config:
        arbitrary_types_allowed = True

    @property
    def InputType(self) -> Any:
        return List[self.bound.InputType]

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        return create_model(self.get_name('Input'), __root__=(List[self.bound.get_input_schema(config)], None))

    @property
    def OutputType(self) -> Type[List[Output]]:
        return List[self.bound.OutputType]

    def get_output_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        schema = self.bound.get_output_schema(config)
        return create_model(self.get_name('Output'), __root__=(List[schema], None))

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self.bound.config_specs

    def get_graph(self, config: Optional[RunnableConfig]=None) -> Graph:
        return self.bound.get_graph(config)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    def _invoke(self, inputs: List[Input], run_manager: CallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> List[Output]:
        return self.bound.batch(inputs, patch_config(config, callbacks=run_manager.get_child()), **kwargs)

    def invoke(self, input: List[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> List[Output]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(self, inputs: List[Input], run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> List[Output]:
        return await self.bound.abatch(inputs, patch_config(config, callbacks=run_manager.get_child()), **kwargs)

    async def ainvoke(self, input: List[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> List[Output]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    async def astream_events(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[StreamEvent]:
        for _ in range(1):
            raise NotImplementedError('RunnableEach does not support astream_events yet.')
            yield