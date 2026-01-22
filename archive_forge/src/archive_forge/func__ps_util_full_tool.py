from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
def _ps_util_full_tool(memory_metric):
    process = psutil.Process(pid)
    try:
        if not hasattr(process, 'memory_full_info'):
            raise NotImplementedError('Backend `{}` requires psutil > 4.0.0'.format(memory_metric))
        meminfo_attr = 'memory_full_info'
        meminfo = getattr(process, meminfo_attr)()
        if not hasattr(meminfo, memory_metric):
            raise NotImplementedError('Metric `{}` not available. For details, see:'.format(memory_metric) + 'https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_info#psutil.Process.memory_full_info')
        mem = getattr(meminfo, memory_metric) / _TWO_20
        if include_children:
            mem += sum([mem for pid, mem in _get_child_memory(process, meminfo_attr, memory_metric)])
        if timestamps:
            return (mem, time.time())
        else:
            return mem
    except psutil.AccessDenied:
        pass