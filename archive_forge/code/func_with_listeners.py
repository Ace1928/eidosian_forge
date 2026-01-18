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
def with_listeners(self, *, on_start: Optional[Listener]=None, on_end: Optional[Listener]=None, on_error: Optional[Listener]=None) -> Runnable[Input, Output]:
    """Bind lifecycle listeners to a Runnable, returning a new Runnable.

        Args:
            on_start: Called before the runnable starts running, with the Run object.
            on_end: Called after the runnable finishes running, with the Run object.
            on_error: Called if the runnable throws an error, with the Run object.

        Returns:
            The Run object contains information about the run, including its id,
            type, input, output, error, start_time, end_time, and any tags or metadata
            added to the run.
        """
    from langchain_core.tracers.root_listeners import RootListenersTracer
    return self.__class__(bound=self.bound, kwargs=self.kwargs, config=self.config, config_factories=[lambda config: {'callbacks': [RootListenersTracer(config=config, on_start=on_start, on_end=on_end, on_error=on_error)]}], custom_input_type=self.custom_input_type, custom_output_type=self.custom_output_type)