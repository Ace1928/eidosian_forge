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
@contextmanager
def tracing_v2_enabled(project_name: Optional[str]=None, *, example_id: Optional[Union[str, UUID]]=None, tags: Optional[List[str]]=None, client: Optional[LangSmithClient]=None) -> Generator[LangChainTracer, None, None]:
    """Instruct LangChain to log all runs in context to LangSmith.

    Args:
        project_name (str, optional): The name of the project.
            Defaults to "default".
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        tags (List[str], optional): The tags to add to the run.
            Defaults to None.
        client (LangSmithClient, optional): The client of the langsmith.
            Defaults to None.

    Returns:
        None

    Example:
        >>> with tracing_v2_enabled():
        ...     # LangChain code will automatically be traced

        You can use this to fetch the LangSmith run URL:

        >>> with tracing_v2_enabled() as cb:
        ...     chain.invoke("foo")
        ...     run_url = cb.get_run_url()
    """
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(example_id=example_id, project_name=project_name, tags=tags, client=client)
    try:
        tracing_v2_callback_var.set(cb)
        yield cb
    finally:
        tracing_v2_callback_var.set(None)