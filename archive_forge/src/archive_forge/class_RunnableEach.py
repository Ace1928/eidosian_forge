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
class RunnableEach(RunnableEachBase[Input, Output]):
    """Runnable that delegates calls to another Runnable
    with each element of the input sequence.

    It allows you to call multiple inputs with the bounded Runnable.

    RunnableEach makes it easy to run multiple inputs for the runnable.
    In the below example, we associate and run three inputs
    with a Runnable:

        .. code-block:: python

            from langchain_core.runnables.base import RunnableEach
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            prompt = ChatPromptTemplate.from_template("Tell me a short joke about
            {topic}")
            model = ChatOpenAI()
            output_parser = StrOutputParser()
            runnable = prompt | model | output_parser
            runnable_each = RunnableEach(bound=runnable)
            output = runnable_each.invoke([{'topic':'Computer Science'},
                                        {'topic':'Art'},
                                        {'topic':'Biology'}])
            print(output)  # noqa: T201
    """

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'runnable']

    def get_name(self, suffix: Optional[str]=None, *, name: Optional[str]=None) -> str:
        name = name or self.name or f'RunnableEach<{self.bound.get_name()}>'
        return super().get_name(suffix, name=name)

    def bind(self, **kwargs: Any) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.bind(**kwargs))

    def with_config(self, config: Optional[RunnableConfig]=None, **kwargs: Any) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.with_config(config, **kwargs))

    def with_listeners(self, *, on_start: Optional[Listener]=None, on_end: Optional[Listener]=None, on_error: Optional[Listener]=None) -> RunnableEach[Input, Output]:
        """
        Bind lifecycle listeners to a Runnable, returning a new Runnable.

        on_start: Called before the runnable starts running, with the Run object.
        on_end: Called after the runnable finishes running, with the Run object.
        on_error: Called if the runnable throws an error, with the Run object.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.
        """
        return RunnableEach(bound=self.bound.with_listeners(on_start=on_start, on_end=on_end, on_error=on_error))