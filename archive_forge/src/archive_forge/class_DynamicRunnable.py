from __future__ import annotations
import enum
import threading
from abc import abstractmethod
from functools import wraps
from typing import (
from weakref import WeakValueDictionary
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
class DynamicRunnable(RunnableSerializable[Input, Output]):
    """Serializable Runnable that can be dynamically configured."""
    default: RunnableSerializable[Input, Output]
    config: Optional[RunnableConfig] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    @property
    def InputType(self) -> Type[Input]:
        return self.default.InputType

    @property
    def OutputType(self) -> Type[Output]:
        return self.default.OutputType

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        runnable, config = self.prepare(config)
        return runnable.get_input_schema(config)

    def get_output_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        runnable, config = self.prepare(config)
        return runnable.get_output_schema(config)

    def get_graph(self, config: Optional[RunnableConfig]=None) -> Graph:
        runnable, config = self.prepare(config)
        return runnable.get_graph(config)

    def with_config(self, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(**{**self.__dict__, 'config': ensure_config(merge_configs(config, kwargs))})

    def prepare(self, config: Optional[RunnableConfig]=None) -> Tuple[Runnable[Input, Output], RunnableConfig]:
        runnable: Runnable[Input, Output] = self
        while isinstance(runnable, DynamicRunnable):
            runnable, config = runnable._prepare(merge_configs(runnable.config, config))
        return (runnable, cast(RunnableConfig, config))

    @abstractmethod
    def _prepare(self, config: Optional[RunnableConfig]=None) -> Tuple[Runnable[Input, Output], RunnableConfig]:
        ...

    def invoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Output:
        runnable, config = self.prepare(config)
        return runnable.invoke(input, config, **kwargs)

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Output:
        runnable, config = self.prepare(config)
        return await runnable.ainvoke(input, config, **kwargs)

    def batch(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, **kwargs: Optional[Any]) -> List[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self.prepare(c) for c in configs]
        if all((p is self.default for p, _ in prepared)):
            return self.default.batch(inputs, [c for _, c in prepared], return_exceptions=return_exceptions, **kwargs)
        if not inputs:
            return []

        def invoke(prepared: Tuple[Runnable[Input, Output], RunnableConfig], input: Input) -> Union[Output, Exception]:
            bound, config = prepared
            if return_exceptions:
                try:
                    return bound.invoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return bound.invoke(input, config, **kwargs)
        if len(inputs) == 1:
            return cast(List[Output], [invoke(prepared[0], inputs[0])])
        with get_executor_for_config(configs[0]) as executor:
            return cast(List[Output], list(executor.map(invoke, prepared, inputs)))

    async def abatch(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]]=None, *, return_exceptions: bool=False, **kwargs: Optional[Any]) -> List[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self.prepare(c) for c in configs]
        if all((p is self.default for p, _ in prepared)):
            return await self.default.abatch(inputs, [c for _, c in prepared], return_exceptions=return_exceptions, **kwargs)
        if not inputs:
            return []

        async def ainvoke(prepared: Tuple[Runnable[Input, Output], RunnableConfig], input: Input) -> Union[Output, Exception]:
            bound, config = prepared
            if return_exceptions:
                try:
                    return await bound.ainvoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await bound.ainvoke(input, config, **kwargs)
        coros = map(ainvoke, prepared, inputs)
        return await gather_with_concurrency(configs[0].get('max_concurrency'), *coros)

    def stream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Iterator[Output]:
        runnable, config = self.prepare(config)
        return runnable.stream(input, config, **kwargs)

    async def astream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[Output]:
        runnable, config = self.prepare(config)
        async for chunk in runnable.astream(input, config, **kwargs):
            yield chunk

    def transform(self, input: Iterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Iterator[Output]:
        runnable, config = self.prepare(config)
        return runnable.transform(input, config, **kwargs)

    async def atransform(self, input: AsyncIterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[Output]:
        runnable, config = self.prepare(config)
        async for chunk in runnable.atransform(input, config, **kwargs):
            yield chunk

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.default, name)
        if callable(attr):

            @wraps(attr)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                for key, arg in kwargs.items():
                    if key == 'config' and (isinstance(arg, dict) and 'configurable' in arg and isinstance(arg['configurable'], dict)):
                        runnable, config = self.prepare(cast(RunnableConfig, arg))
                        kwargs = {**kwargs, 'config': config}
                        return getattr(runnable, name)(*args, **kwargs)
                for idx, arg in enumerate(args):
                    if isinstance(arg, dict) and 'configurable' in arg and isinstance(arg['configurable'], dict):
                        runnable, config = self.prepare(cast(RunnableConfig, arg))
                        argsl = list(args)
                        argsl[idx] = config
                        return getattr(runnable, name)(*argsl, **kwargs)
                if self.config:
                    runnable, config = self.prepare()
                    return getattr(runnable, name)(*args, **kwargs)
                return attr(*args, **kwargs)
            return wrapper
        else:
            return attr