from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
@contextlib.contextmanager
def tracing_context(*, project_name: Optional[str]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, parent: Optional[Union[run_trees.RunTree, Mapping, str]]=None, **kwargs: Any) -> Generator[None, None, None]:
    """Set the tracing context for a block of code."""
    if kwargs:
        warnings.warn(f'Unrecognized keyword arguments: {kwargs}.', DeprecationWarning)
    current_context = get_tracing_context()
    parent_run = _get_parent_run({'parent': parent or kwargs.get('parent_run')})
    if parent_run is not None:
        tags = sorted(set(tags or []) | set(parent_run.tags or []))
        metadata = {**parent_run.metadata, **(metadata or {})}
    _set_tracing_context({'parent': parent_run, 'project_name': project_name, 'tags': tags, 'metadata': metadata})
    try:
        yield
    finally:
        _set_tracing_context(current_context)