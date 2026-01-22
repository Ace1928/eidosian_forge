from __future__ import annotations
import inspect
import uuid
import warnings
from abc import abstractmethod
from functools import partial
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.callbacks.manager import (
from langchain_core.load.serializable import Serializable
from langchain_core.prompts import (
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
from langchain_core.runnables.config import run_in_executor
def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
    """Convert tool input to pydantic model."""
    args, kwargs = super()._to_args_and_kwargs(tool_input)
    all_args = list(args) + list(kwargs.values())
    if len(all_args) != 1:
        raise ToolException(f'Too many arguments to single-input tool {self.name}.\n                Consider using StructuredTool instead. Args: {all_args}')
    return (tuple(all_args), {})