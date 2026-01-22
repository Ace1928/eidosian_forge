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
class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that copies the context to the child thread."""

    def submit(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        """Submit a function to the executor.

        Args:
            func (Callable[..., T]): The function to submit.
            *args (Any): The positional arguments to the function.
            **kwargs (Any): The keyword arguments to the function.

        Returns:
            Future[T]: The future for the function.
        """
        return super().submit(cast(Callable[..., T], partial(copy_context().run, func, *args, **kwargs)))

    def map(self, fn: Callable[..., T], *iterables: Iterable[Any], timeout: float | None=None, chunksize: int=1) -> Iterator[T]:
        contexts = [copy_context() for _ in range(len(iterables[0]))]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)
        return super().map(_wrapped_fn, *iterables, timeout=timeout, chunksize=chunksize)