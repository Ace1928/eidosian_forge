from __future__ import annotations
import asyncio
import functools
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import (
from uuid import UUID
from langsmith.run_helpers import get_run_tree_context
from tenacity import RetryCallState
from langchain_core.callbacks.base import (
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils.env import env_var_is_set
def _run_coros(coros: List[Coroutine[Any, Any, Any]]) -> None:
    if hasattr(asyncio, 'Runner'):
        with asyncio.Runner() as runner:
            for coro in coros:
                try:
                    runner.run(coro)
                except Exception as e:
                    logger.warning(f'Error in callback coroutine: {repr(e)}')
            while (pending := asyncio.all_tasks(runner.get_loop())):
                runner.run(asyncio.wait(pending))
    else:
        for coro in coros:
            try:
                asyncio.run(coro)
            except Exception as e:
                logger.warning(f'Error in callback coroutine: {repr(e)}')