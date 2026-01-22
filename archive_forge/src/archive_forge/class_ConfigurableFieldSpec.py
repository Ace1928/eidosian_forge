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
class ConfigurableFieldSpec(NamedTuple):
    """Field that can be configured by the user. It is a specification of a field."""
    id: str
    annotation: Any
    name: Optional[str] = None
    description: Optional[str] = None
    default: Any = None
    is_shared: bool = False
    dependencies: Optional[List[str]] = None