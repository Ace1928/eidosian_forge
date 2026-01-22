from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import UUID
from langsmith import utils as ls_utils
from langsmith.run_helpers import get_run_tree_context
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.schemas import TracerSessionV1
from langchain_core.utils.env import env_var_is_set
def _get_tracer_project() -> str:
    run_tree = get_run_tree_context()
    return getattr(run_tree, 'session_name', getattr(tracing_v2_callback_var.get(), 'project', str(ls_utils.get_tracer_project())))