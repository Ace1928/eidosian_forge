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
def trace_memory_usage(self, frame, event, arg):
    """Callback for sys.settrace"""
    if frame.f_code in self.code_map:
        if event == 'call':
            self.prevlines.append(frame.f_lineno)
        elif event == 'line':
            self.code_map.trace(frame.f_code, self.prevlines[-1], self.prev_lineno)
            self.prev_lineno = self.prevlines[-1]
            self.prevlines[-1] = frame.f_lineno
        elif event == 'return':
            lineno = self.prevlines.pop()
            self.code_map.trace(frame.f_code, lineno, self.prev_lineno)
            self.prev_lineno = lineno
    if self._original_trace_function is not None:
        self._original_trace_function(frame, event, arg)
    return self.trace_memory_usage