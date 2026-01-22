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
class TimeStamper:
    """ A profiler that just records start and end execution times for
    any decorated function.
    """

    def __init__(self, backend, include_children=False):
        self.functions = {}
        self.backend = backend
        self.include_children = include_children
        self.current_stack_level = -1
        self.stack = {}

    def __call__(self, func=None, precision=None):
        if func is not None:
            if not callable(func):
                raise ValueError('Value must be callable')
            self.add_function(func)
            f = self.wrap_function(func)
            f.__module__ = func.__module__
            f.__name__ = func.__name__
            f.__doc__ = func.__doc__
            f.__dict__.update(getattr(func, '__dict__', {}))
            return f
        else:

            def inner_partial(f):
                return self.__call__(f, precision=precision)
            return inner_partial

    def timestamp(self, name='<block>'):
        """Returns a context manager for timestamping a block of code."""
        func = lambda x: x
        func.__module__ = ''
        func.__name__ = name
        self.add_function(func)
        timestamps = []
        self.functions[func].append(timestamps)
        try:
            filename = inspect.getsourcefile(func)
        except TypeError:
            filename = '<unknown>'
        return _TimeStamperCM(timestamps, filename, self.backend, timestamper=self, func=func)

    def add_function(self, func):
        if func not in self.functions:
            self.functions[func] = []
            self.stack[func] = []

    def wrap_function(self, func):
        """ Wrap a function to timestamp it.
        """

        def f(*args, **kwds):
            try:
                filename = inspect.getsourcefile(func)
            except TypeError:
                filename = '<unknown>'
            timestamps = [_get_memory(os.getpid(), self.backend, timestamps=True, include_children=self.include_children, filename=filename)]
            self.functions[func].append(timestamps)
            try:
                with self.call_on_stack(func, *args, **kwds) as result:
                    return result
            finally:
                timestamps.append(_get_memory(os.getpid(), self.backend, timestamps=True, include_children=self.include_children, filename=filename))
        return f

    @contextmanager
    def call_on_stack(self, func, *args, **kwds):
        self.current_stack_level += 1
        self.stack[func].append(self.current_stack_level)
        yield func(*args, **kwds)
        self.current_stack_level -= 1

    def show_results(self, stream=None):
        if stream is None:
            stream = sys.stdout
        for func, timestamps in self.functions.items():
            function_name = '%s.%s' % (func.__module__, func.__name__)
            for ts, level in zip(timestamps, self.stack[func]):
                stream.write('FUNC %s %.4f %.4f %.4f %.4f %d\n' % ((function_name,) + ts[0] + ts[1] + (level,)))