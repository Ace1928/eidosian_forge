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
def tracemalloc_tool():
    stat = next(filter(lambda item: str(item).startswith(filename), tracemalloc.take_snapshot().statistics('filename')))
    mem = stat.size / _TWO_20
    if timestamps:
        return (mem, time.time())
    else:
        return mem