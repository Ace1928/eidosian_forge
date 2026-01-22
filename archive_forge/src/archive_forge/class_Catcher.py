import builtins
import contextlib
import functools
import logging
import re
import sys
import warnings
from collections import namedtuple
from inspect import isclass, iscoroutinefunction, isgeneratorfunction
from multiprocessing import current_process, get_context
from multiprocessing.context import BaseContext
from os.path import basename, splitext
from threading import current_thread
from . import _asyncio_loop, _colorama, _defaults, _filters
from ._better_exceptions import ExceptionFormatter
from ._colorizer import Colorizer
from ._contextvars import ContextVar
from ._datetime import aware_now
from ._error_interceptor import ErrorInterceptor
from ._file_sink import FileSink
from ._get_frame import get_frame
from ._handler import Handler
from ._locks_machinery import create_logger_lock
from ._recattrs import RecordException, RecordFile, RecordLevel, RecordProcess, RecordThread
from ._simple_sinks import AsyncSink, CallableSink, StandardSink, StreamSink
class Catcher:

    def __init__(self, from_decorator):
        self._from_decorator = from_decorator

    def __enter__(self):
        return None

    def __exit__(self, type_, value, traceback_):
        if type_ is None:
            return
        if not issubclass(type_, exception):
            return False
        if exclude is not None and issubclass(type_, exclude):
            return False
        from_decorator = self._from_decorator
        _, depth, _, *options = logger._options
        if from_decorator:
            depth += 1
        catch_options = [(type_, value, traceback_), depth, True] + options
        logger._log(level, from_decorator, catch_options, message, (), {})
        if onerror is not None:
            onerror(value)
        return not reraise

    def __call__(self, function):
        if isclass(function):
            raise TypeError("Invalid object decorated with 'catch()', it must be a function, not a class (tried to wrap '%s')" % function.__name__)
        catcher = Catcher(True)
        if iscoroutinefunction(function):

            async def catch_wrapper(*args, **kwargs):
                with catcher:
                    return await function(*args, **kwargs)
                return default
        elif isgeneratorfunction(function):

            def catch_wrapper(*args, **kwargs):
                with catcher:
                    return (yield from function(*args, **kwargs))
                return default
        else:

            def catch_wrapper(*args, **kwargs):
                with catcher:
                    return function(*args, **kwargs)
                return default
        functools.update_wrapper(catch_wrapper, function)
        return catch_wrapper