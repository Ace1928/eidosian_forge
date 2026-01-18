from __future__ import annotations
import os
import abc
import sys
import anyio
import inspect
import asyncio
import functools
import subprocess
import contextvars
import anyio.from_thread
from concurrent import futures
from anyio._core._eventloop import threadlocals
from lazyops.libs.proxyobj import ProxyObject
from typing import Callable, Coroutine, Any, Union, List, Set, Tuple, TypeVar, Optional, Generator, Awaitable, Iterable, AsyncGenerator, Dict
def threadpool_task(self, func: Callable, *args, task_callback: Optional[Callable[..., RT]]=None, task_callback_args: Optional[Tuple]=None, task_callback_kwargs: Optional[Dict]=None, **kwargs) -> futures.Future[RT]:
    """
        Creates a threadpool task
        """
    task = self.pool.submit(func, *args, **kwargs)
    self.add_task(task, task_callback, callback_args=task_callback_args, callback_kwargs=task_callback_kwargs)
    return task