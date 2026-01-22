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
class _TraceableContainer(TypedDict, total=False):
    """Typed response when initializing a run a traceable."""
    new_run: Optional[run_trees.RunTree]
    project_name: Optional[str]
    outer_project: Optional[str]
    outer_metadata: Optional[Dict[str, Any]]
    outer_tags: Optional[List[str]]
    on_end: Optional[Callable[[run_trees.RunTree], Any]]
    context: contextvars.Context