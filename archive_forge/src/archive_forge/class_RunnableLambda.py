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
class RunnableLambda(Runnable[Input, Output]):
    """RunnableLambda converts a python callable into a Runnable.

    Wrapping a callable in a RunnableLambda makes the callable usable
    within either a sync or async context.

    RunnableLambda can be composed as any other Runnable and provides
    seamless integration with LangChain tracing.

    Examples:

        .. code-block:: python

            # This is a RunnableLambda
            from langchain_core.runnables import RunnableLambda

            def add_one(x: int) -> int:
                return x + 1

            runnable = RunnableLambda(add_one)

            runnable.invoke(1) # returns 2
            runnable.batch([1, 2, 3]) # returns [2, 3, 4]

            # Async is supported by default by delegating to the sync implementation
            await runnable.ainvoke(1) # returns 2
            await runnable.abatch([1, 2, 3]) # returns [2, 3, 4]


            # Alternatively, can provide both synd and sync implementations
            async def add_one_async(x: int) -> int:
                return x + 1

            runnable = RunnableLambda(add_one, afunc=add_one_async)
            runnable.invoke(1) # Uses add_one
            await runnable.ainvoke(1) # Uses add_one_async
    """

    def __init__(self, func: Union[Union[Callable[[Input], Output], Callable[[Input], Iterator[Output]], Callable[[Input, RunnableConfig], Output], Callable[[Input, CallbackManagerForChainRun], Output], Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output]], Union[Callable[[Input], Awaitable[Output]], Callable[[Input], AsyncIterator[Output]], Callable[[Input, RunnableConfig], Awaitable[Output]], Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]], Callable[[Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]]]], afunc: Optional[Union[Callable[[Input], Awaitable[Output]], Callable[[Input], AsyncIterator[Output]], Callable[[Input, RunnableConfig], Awaitable[Output]], Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]], Callable[[Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]]]]=None, name: Optional[str]=None) -> None:
        """Create a RunnableLambda from a callable, and async callable or both.

        Accepts both sync and async variants to allow providing efficient
        implementations for sync and async execution.

        Args:
            func: Either sync or async callable
            afunc: An async callable that takes an input and returns an output.
        """
        if afunc is not None:
            self.afunc = afunc
            func_for_name: Callable = afunc
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            if afunc is not None:
                raise TypeError('Func was provided as a coroutine function, but afunc was also provided. If providing both, func should be a regular function to avoid ambiguity.')
            self.afunc = func
            func_for_name = func
        elif callable(func):
            self.func = cast(Callable[[Input], Output], func)
            func_for_name = func
        else:
            raise TypeError(f'Expected a callable type for `func`.Instead got an unsupported type: {type(func)}')
        try:
            if name is not None:
                self.name = name
            elif func_for_name.__name__ != '<lambda>':
                self.name = func_for_name.__name__
        except AttributeError:
            pass

    @property
    def InputType(self) -> Any:
        """The type of the input to this runnable."""
        func = getattr(self, 'func', None) or getattr(self, 'afunc')
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return first_param.annotation
            else:
                return Any
        except ValueError:
            return Any

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        """The pydantic schema for the input to this runnable."""
        func = getattr(self, 'func', None) or getattr(self, 'afunc')
        if isinstance(func, itemgetter):
            items = str(func).replace('operator.itemgetter(', '')[:-1].split(', ')
            if all((item[0] == "'" and item[-1] == "'" and (len(item) > 2) for item in items)):
                return create_model(self.get_name('Input'), **{item[1:-1]: (Any, None) for item in items})
            else:
                return create_model(self.get_name('Input'), __root__=(List[Any], None))
        if self.InputType != Any:
            return super().get_input_schema(config)
        if (dict_keys := get_function_first_arg_dict_keys(func)):
            return create_model(self.get_name('Input'), **{key: (Any, None) for key in dict_keys})
        return super().get_input_schema(config)

    @property
    def OutputType(self) -> Any:
        """The type of the output of this runnable as a type annotation."""
        func = getattr(self, 'func', None) or getattr(self, 'afunc')
        try:
            sig = inspect.signature(func)
            if sig.return_annotation != inspect.Signature.empty:
                if getattr(sig.return_annotation, '__origin__', None) in (collections.abc.Iterator, collections.abc.AsyncIterator):
                    return getattr(sig.return_annotation, '__args__', (Any,))[0]
                return sig.return_annotation
            else:
                return Any
        except ValueError:
            return Any

    @property
    def deps(self) -> List[Runnable]:
        """The dependencies of this runnable."""
        if hasattr(self, 'func'):
            objects = get_function_nonlocals(self.func)
        elif hasattr(self, 'afunc'):
            objects = get_function_nonlocals(self.afunc)
        else:
            objects = []
        deps: List[Runnable] = []
        for obj in objects:
            if isinstance(obj, Runnable):
                deps.append(obj)
            elif isinstance(getattr(obj, '__self__', None), Runnable):
                deps.append(obj.__self__)
        return deps

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs((spec for dep in self.deps for spec in dep.config_specs))

    def get_graph(self, config: RunnableConfig | None=None) -> Graph:
        if (deps := self.deps):
            graph = Graph()
            input_node = graph.add_node(self.get_input_schema(config))
            output_node = graph.add_node(self.get_output_schema(config))
            for dep in deps:
                dep_graph = dep.get_graph()
                dep_graph.trim_first_node()
                dep_graph.trim_last_node()
                if not dep_graph:
                    graph.add_edge(input_node, output_node)
                else:
                    graph.extend(dep_graph)
                    dep_first_node = dep_graph.first_node()
                    if not dep_first_node:
                        raise ValueError(f'Runnable {dep} has no first node')
                    dep_last_node = dep_graph.last_node()
                    if not dep_last_node:
                        raise ValueError(f'Runnable {dep} has no last node')
                    graph.add_edge(input_node, dep_first_node)
                    graph.add_edge(dep_last_node, output_node)
        else:
            graph = super().get_graph(config)
        return graph

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RunnableLambda):
            if hasattr(self, 'func') and hasattr(other, 'func'):
                return self.func == other.func
            elif hasattr(self, 'afunc') and hasattr(other, 'afunc'):
                return self.afunc == other.afunc
            else:
                return False
        else:
            return False

    def __repr__(self) -> str:
        """A string representation of this runnable."""
        if hasattr(self, 'func') and isinstance(self.func, itemgetter):
            return f'RunnableLambda({str(self.func)[len('operator.'):]})'
        elif hasattr(self, 'func'):
            return f'RunnableLambda({get_lambda_source(self.func) or '...'})'
        elif hasattr(self, 'afunc'):
            return f'RunnableLambda(afunc={get_lambda_source(self.afunc) or '...'})'
        else:
            return 'RunnableLambda(...)'

    def _invoke(self, input: Input, run_manager: CallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Output:
        if inspect.isgeneratorfunction(self.func):
            output: Optional[Output] = None
            for chunk in call_func_with_variable_args(cast(Callable[[Input], Iterator[Output]], self.func), input, config, run_manager, **kwargs):
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk
                    except TypeError:
                        output = chunk
        else:
            output = call_func_with_variable_args(self.func, input, config, run_manager, **kwargs)
        if isinstance(output, Runnable):
            recursion_limit = config['recursion_limit']
            if recursion_limit <= 0:
                raise RecursionError(f'Recursion limit reached when invoking {self} with input {input}.')
            output = output.invoke(input, patch_config(config, callbacks=run_manager.get_child(), recursion_limit=recursion_limit - 1))
        return cast(Output, output)

    async def _ainvoke(self, input: Input, run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Output:
        if hasattr(self, 'afunc'):
            afunc = self.afunc
        else:
            if inspect.isgeneratorfunction(self.func):

                def func(input: Input, run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Output:
                    output: Optional[Output] = None
                    for chunk in call_func_with_variable_args(cast(Callable[[Input], Iterator[Output]], self.func), input, config, run_manager.get_sync(), **kwargs):
                        if output is None:
                            output = chunk
                        else:
                            try:
                                output = output + chunk
                            except TypeError:
                                output = chunk
                    return cast(Output, output)
            else:

                def func(input: Input, run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Output:
                    return call_func_with_variable_args(self.func, input, config, run_manager.get_sync(), **kwargs)

            @wraps(func)
            async def f(*args, **kwargs):
                return await run_in_executor(config, func, *args, **kwargs)
            afunc = f
        if inspect.isasyncgenfunction(afunc):
            output: Optional[Output] = None
            async for chunk in cast(AsyncIterator[Output], acall_func_with_variable_args(cast(Callable, afunc), input, config, run_manager, **kwargs)):
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk
                    except TypeError:
                        output = chunk
        else:
            output = await acall_func_with_variable_args(cast(Callable, afunc), input, config, run_manager, **kwargs)
        if isinstance(output, Runnable):
            recursion_limit = config['recursion_limit']
            if recursion_limit <= 0:
                raise RecursionError(f'Recursion limit reached when invoking {self} with input {input}.')
            output = await output.ainvoke(input, patch_config(config, callbacks=run_manager.get_child(), recursion_limit=recursion_limit - 1))
        return cast(Output, output)

    def _config(self, config: Optional[RunnableConfig], callable: Callable[..., Any]) -> RunnableConfig:
        return ensure_config(config)

    def invoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Output:
        """Invoke this runnable synchronously."""
        if hasattr(self, 'func'):
            return self._call_with_config(self._invoke, input, self._config(config, self.func), **kwargs)
        else:
            raise TypeError('Cannot invoke a coroutine function synchronously.Use `ainvoke` instead.')

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Output:
        """Invoke this runnable asynchronously."""
        the_func = self.afunc if hasattr(self, 'afunc') else self.func
        return await self._acall_with_config(self._ainvoke, input, self._config(config, the_func), **kwargs)

    def _transform(self, input: Iterator[Input], run_manager: CallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Iterator[Output]:
        final: Optional[Input] = None
        for ichunk in input:
            if final is None:
                final = adapt_first_streaming_chunk(ichunk)
            else:
                try:
                    final = final + ichunk
                except TypeError:
                    final = ichunk
        if inspect.isgeneratorfunction(self.func):
            output: Optional[Output] = None
            for chunk in call_func_with_variable_args(self.func, cast(Input, final), config, run_manager, **kwargs):
                yield chunk
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk
                    except TypeError:
                        output = chunk
        else:
            output = call_func_with_variable_args(self.func, cast(Input, final), config, run_manager, **kwargs)
        if isinstance(output, Runnable):
            recursion_limit = config['recursion_limit']
            if recursion_limit <= 0:
                raise RecursionError(f'Recursion limit reached when invoking {self} with input {final}.')
            for chunk in output.stream(final, patch_config(config, callbacks=run_manager.get_child(), recursion_limit=recursion_limit - 1)):
                yield chunk
        elif not inspect.isgeneratorfunction(self.func):
            yield cast(Output, output)

    def transform(self, input: Iterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Iterator[Output]:
        if hasattr(self, 'func'):
            for output in self._transform_stream_with_config(input, self._transform, self._config(config, self.func), **kwargs):
                yield output
        else:
            raise TypeError('Cannot stream a coroutine function synchronously.Use `astream` instead.')

    def stream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    async def _atransform(self, input: AsyncIterator[Input], run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> AsyncIterator[Output]:
        final: Optional[Input] = None
        async for ichunk in input:
            if final is None:
                final = adapt_first_streaming_chunk(ichunk)
            else:
                try:
                    final = final + ichunk
                except TypeError:
                    final = ichunk
        if hasattr(self, 'afunc'):
            afunc = self.afunc
        else:
            if inspect.isgeneratorfunction(self.func):
                raise TypeError('Cannot stream from a generator function asynchronously.Use .stream() instead.')

            def func(input: Input, run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig, **kwargs: Any) -> Output:
                return call_func_with_variable_args(self.func, input, config, run_manager.get_sync(), **kwargs)

            @wraps(func)
            async def f(*args, **kwargs):
                return await run_in_executor(config, func, *args, **kwargs)
            afunc = f
        if inspect.isasyncgenfunction(afunc):
            output: Optional[Output] = None
            async for chunk in cast(AsyncIterator[Output], acall_func_with_variable_args(cast(Callable, afunc), cast(Input, final), config, run_manager, **kwargs)):
                yield chunk
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk
                    except TypeError:
                        output = chunk
        else:
            output = await acall_func_with_variable_args(cast(Callable, afunc), cast(Input, final), config, run_manager, **kwargs)
        if isinstance(output, Runnable):
            recursion_limit = config['recursion_limit']
            if recursion_limit <= 0:
                raise RecursionError(f'Recursion limit reached when invoking {self} with input {final}.')
            async for chunk in output.astream(final, patch_config(config, callbacks=run_manager.get_child(), recursion_limit=recursion_limit - 1)):
                yield chunk
        elif not inspect.isasyncgenfunction(afunc):
            yield cast(Output, output)

    async def atransform(self, input: AsyncIterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[Output]:
        async for output in self._atransform_stream_with_config(input, self._atransform, self._config(config, self.afunc if hasattr(self, 'afunc') else self.func), **kwargs):
            yield output

    async def astream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Optional[Any]) -> AsyncIterator[Output]:

        async def input_aiter() -> AsyncIterator[Input]:
            yield input
        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk