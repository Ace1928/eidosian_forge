from __future__ import annotations
import asyncio
import uuid
import warnings
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
from typing import (
from typing_extensions import ParamSpec, TypedDict
from langchain_core.runnables.utils import (
class RunnableConfig(TypedDict, total=False):
    """Configuration for a Runnable."""
    tags: List[str]
    '\n    Tags for this call and any sub-calls (eg. a Chain calling an LLM).\n    You can use these to filter calls.\n    '
    metadata: Dict[str, Any]
    '\n    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).\n    Keys should be strings, values should be JSON-serializable.\n    '
    callbacks: Callbacks
    '\n    Callbacks for this call and any sub-calls (eg. a Chain calling an LLM).\n    Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.\n    '
    run_name: str
    '\n    Name for the tracer run for this call. Defaults to the name of the class.\n    '
    max_concurrency: Optional[int]
    "\n    Maximum number of parallel calls to make. If not provided, defaults to \n    ThreadPoolExecutor's default.\n    "
    recursion_limit: int
    '\n    Maximum number of times a call can recurse. If not provided, defaults to 25.\n    '
    configurable: Dict[str, Any]
    '\n    Runtime values for attributes previously made configurable on this Runnable,\n    or sub-Runnables, through .configurable_fields() or .configurable_alternatives().\n    Check .output_schema() for a description of the attributes that have been made \n    configurable.\n    '
    run_id: Optional[uuid.UUID]
    '\n    Unique identifier for the tracer run for this call. If not provided, a new UUID\n        will be generated.\n    '