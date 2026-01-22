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
@lru_cache(maxsize=256)
def _create_model_cached(__model_name: str, **field_definitions: Any) -> Type[BaseModel]:
    return _create_model_base(__model_name, __config__=_SchemaConfig, **field_definitions)