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
class RunnableGenerator(Runnable[Input, Output]):
    """Runnable that runs a generator function.

    RunnableGenerators can be instantiated directly or by using a generator within
    a sequence.

    RunnableGenerators can be used to implement custom behavior, such as custom output
    parsers, while preserving streaming capabilities. Given a generator function with
    a signature Iterator[A] -> Iterator[B], wrapping it in a RunnableGenerator allows
    it to emit output chunks as soon as they are streamed in from the previous step.

    Note that if a generator function has a signature A -> Iterator[B], such that it
    requires its input from the previous step to be completed before emitting chunks
    (e.g., most LLMs need the entire prompt available to start generating), it can
    instead be wrapped in a RunnableLambda.

    Here is an example to show the basic mechanics of a RunnableGenerator:

        .. code-block:: python

            from typing import Any, AsyncIterator, Iterator

            from langchain_core.runnables import RunnableGenerator


            def gen(input: Iterator[Any]) -> Iterator[str]:
                for token in ["Have", " a", " nice", " day"]:
                    yield token


            runnable = RunnableGenerator(gen)
            runnable.invoke(None)  # "Have a nice day"
            list(runnable.stream(None))  # ["Have", " a", " nice", " day"]
            runnable.batch([None, None])  # ["Have a nice day", "Have a nice day"]


            # Async version:
            async def agen(input: AsyncIterator[Any]) -> AsyncIterator[str]:
                for token in ["Have", " a", " nice", " day"]:
                    yield token

            runnable = RunnableGenerator(agen)
            await runnable.ainvoke(None)  # "Have a nice day"
            [p async for p in runnable.astream(None)] # ["Have", " a", " nice", " day"]

    RunnableGenerator makes it easy to implement custom behavior within a streaming
    context. Below we show an example:
        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableGenerator, RunnableLambda
            from langchain_openai import ChatOpenAI
            from langchain_core.output_parsers import StrOutputParser


            model = ChatOpenAI()
            chant_chain = (
                ChatPromptTemplate.from_template("Give me a 3 word chant about {topic}")
                | model
                | StrOutputParser()
            )

            def character_generator(input: Iterator[str]) -> Iterator[str]:
                for token in input:
                    if "," in token or "." in token:
                        yield "👏" + token
                    else:
                        yield token


            runnable = chant_chain | character_generator
            assert type(runnable.last) is RunnableGenerator
            "".join(runnable.stream({"topic": "waste"})) # Reduce👏, Reuse👏, Recycle👏.

            # Note that RunnableLambda can be used to delay streaming of one step in a
            # sequence until the previous step is finished:
            def reverse_generator(input: str) -> Iterator[str]:
                # Yield characters of input in reverse order.
                for character in input[::-1]:
                    yield character

            runnable = chant_chain | RunnableLambda(reverse_generator)
            "".join(runnable.stream({"topic": "waste"}))  # ".elcycer ,esuer ,ecudeR"
    """

    def __init__(self, transform: Union[Callable[[Iterator[Input]], Iterator[Output]], Callable[[AsyncIterator[Input]], AsyncIterator[Output]]], atransform: Optional[Callable[[AsyncIterator[Input]], AsyncIterator[Output]]]=None) -> None:
        if atransform is not None:
            self._atransform = atransform
            func_for_name: Callable = atransform
        if inspect.isasyncgenfunction(transform):
            self._atransform = transform
            func_for_name = transform
        elif inspect.isgeneratorfunction(transform):
            self._transform = transform
            func_for_name = transform
        else:
            raise TypeError(f'Expected a generator function type for `transform`.Instead got an unsupported type: {type(transform)}')
        try:
            self.name = func_for_name.__name__
        except AttributeError:
            pass

    @property
    def InputType(self) -> Any:
        func = getattr(self, '_transform', None) or getattr(self, '_atransform')
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return getattr(first_param.annotation, '__args__', (Any,))[0]
            else:
                return Any
        except ValueError:
            return Any

    @property
    def OutputType(self) -> Any:
        func = getattr(self, '_transform', None) or getattr(self, '_atransform')
        try:
            sig = inspect.signature(func)
            return getattr(sig.return_annotation, '__args__', (Any,))[0] if sig.return_annotation != inspect.Signature.empty else Any
        except ValueError:
            return Any

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RunnableGenerator):
            if hasattr(self, '_transform') and hasattr(other, '_transform'):
                return self._transform == other._transform
            elif hasattr(self, '_atransform') and hasattr(other, '_atransform'):
                return self._atransform == other._atransform
            else:
                return False
        else:
            return False

    def __repr__(self) -> str:
        return f'RunnableGenerator({self.name})'

    def transform(self, input: Iterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Output]:
        if not hasattr(self, '_transform'):
            raise NotImplementedError(f'{repr(self)} only supports async methods.')
        return self._transform_stream_with_config(input, self._transform, config, **kwargs)

    def stream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    def invoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Output:
        final = None
        for output in self.stream(input, config, **kwargs):
            if final is None:
                final = output
            else:
                final = final + output
        return cast(Output, final)

    def atransform(self, input: AsyncIterator[Input], config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Output]:
        if not hasattr(self, '_atransform'):
            raise NotImplementedError(f'{repr(self)} only supports sync methods.')
        return self._atransform_stream_with_config(input, self._atransform, config, **kwargs)

    def astream(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[Output]:

        async def input_aiter() -> AsyncIterator[Input]:
            yield input
        return self.atransform(input_aiter(), config, **kwargs)

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Output:
        final = None
        async for output in self.astream(input, config, **kwargs):
            if final is None:
                final = output
            else:
                final = final + output
        return cast(Output, final)