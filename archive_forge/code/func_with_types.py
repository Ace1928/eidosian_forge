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
def with_types(self, input_type: Optional[Union[Type[Input], BaseModel]]=None, output_type: Optional[Union[Type[Output], BaseModel]]=None) -> Runnable[Input, Output]:
    return self.__class__(bound=self.bound, kwargs=self.kwargs, config=self.config, custom_input_type=input_type if input_type is not None else self.custom_input_type, custom_output_type=output_type if output_type is not None else self.custom_output_type)