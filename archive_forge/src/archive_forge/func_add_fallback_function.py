import os
import re
import json
import socket
import contextlib
import functools
from lazyops.utils.helpers import is_coro_func
from lazyops.utils.logs import default_logger as logger
from typing import Optional, Dict, Any, Union, Callable, List, Tuple, TYPE_CHECKING
from aiokeydb.v2.types import BaseSettings, validator, lazyproperty, KeyDBUri
from aiokeydb.v2.types.static import TaskType
from aiokeydb.v2.serializers import SerializerType
from aiokeydb.v2.utils.queue import run_in_executor
from aiokeydb.v2.utils.cron import validate_cron_schedule
def add_fallback_function(self, verbose: Optional[bool]=None, silenced: Optional[bool]=None, method='apply', timeout: Optional[int]=None, suppressed_exceptions: Optional[list]=None, failed_results: Optional[list]=None, queue_func: Optional[Union[Callable, 'TaskQueue']]=None, silenced_stages: Optional[List[str]]=None, disabled: Optional[bool]=False, **kwargs):
    """
        Creates a fallback function for the worker.

        - attempts to apply the function to the queue, if it fails, it will
        attempt to run the function locally.
        """
    if not suppressed_exceptions:
        suppressed_exceptions = [Exception]
    if verbose is None:
        verbose = self.debug_enabled
    if timeout is None:
        timeout = self.job_timeout

    def decorator(func: Callable):
        if disabled:
            return func
        self.tasks.functions.append(func)
        name = func.__name__
        if verbose:
            logger.info(f'Registered fallback function {name}')
        if silenced is True or silenced_stages:
            self.add_function_to_silenced(name, silenced_stages=silenced_stages)

        @functools.wraps(func)
        async def wrapper(**kwargs):
            with contextlib.suppress(*suppressed_exceptions):
                queue = self.get_queue_func(queue_func)
                res = await getattr(queue, method)(name, timeout=timeout, **kwargs)
                if failed_results and res in failed_results:
                    raise ValueError(res)
                return res
            return await func(ctx=None, **kwargs)
        return wrapper
    return decorator