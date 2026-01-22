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
@staticmethod
def _configure_run_tree(callback_manager: Any) -> Optional[run_trees.RunTree]:
    run_tree: Optional[run_trees.RunTree] = None
    if isinstance(callback_manager, (CallbackManager, AsyncCallbackManager)):
        lc_tracers = [handler for handler in callback_manager.handlers if isinstance(handler, LangChainTracer)]
        if lc_tracers:
            lc_tracer = lc_tracers[0]
            run_tree = run_trees.RunTree(id=callback_manager.parent_run_id, session_name=lc_tracer.project_name, name='Wrapping', run_type='chain', inputs={}, tags=callback_manager.tags, extra={'metadata': callback_manager.metadata})
    return run_tree