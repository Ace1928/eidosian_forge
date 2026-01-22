from __future__ import annotations
import ast
import asyncio
import inspect
import textwrap
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
from langchain_core.pydantic_v1 import BaseConfig, BaseModel
from langchain_core.pydantic_v1 import create_model as _create_model_base
from langchain_core.runnables.schema import StreamEvent
def adapt_first_streaming_chunk(chunk: Any) -> Any:
    """This might transform the first chunk of a stream into an AddableDict."""
    if isinstance(chunk, dict) and (not isinstance(chunk, AddableDict)):
        return AddableDict(chunk)
    else:
        return chunk