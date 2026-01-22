from __future__ import print_function, division, absolute_import
import asyncio
import concurrent.futures
import contextlib
import time
from uuid import uuid4
import weakref
from .parallel import parallel_config
from .parallel import AutoBatchingMixin, ParallelBackendBase
def _make_tasks_summary(tasks):
    """Summarize of list of (func, args, kwargs) function calls"""
    unique_funcs = {func for func, args, kwargs in tasks}
    if len(unique_funcs) == 1:
        mixed = False
    else:
        mixed = True
    return (len(tasks), mixed, _funcname(tasks))