from __future__ import annotations
import codecs
import functools
import inspect
import os
import re
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping, Set
from contextlib import contextmanager, nullcontext, suppress
from datetime import datetime, timedelta
from errno import ENOENT
from functools import lru_cache, wraps
from importlib import import_module
from numbers import Integral, Number
from operator import add
from threading import Lock
from typing import Any, Callable, ClassVar, Literal, TypeVar, cast, overload
from weakref import WeakValueDictionary
import tlz as toolz
from dask import config
from dask.core import get_deps
from dask.typing import no_default
class shorten_traceback:
    """Context manager that removes irrelevant stack elements from traceback.

    * omits frames from modules that match `admin.traceback.shorten`
    * always keeps the first and last frame.
    """
    __slots__ = ()

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None:
        if exc_val and exc_tb:
            exc_val.__traceback__ = self.shorten(exc_tb)

    @staticmethod
    def shorten(exc_tb: types.TracebackType) -> types.TracebackType:
        paths = config.get('admin.traceback.shorten')
        if not paths:
            return exc_tb
        exp = re.compile('.*(' + '|'.join(paths) + ')')
        curr: types.TracebackType | None = exc_tb
        prev: types.TracebackType | None = None
        while curr:
            if prev is None:
                prev = curr
            elif not curr.tb_next:
                prev.tb_next = curr
                prev = prev.tb_next
            elif not exp.match(curr.tb_frame.f_code.co_filename):
                prev.tb_next = curr
                prev = curr
            curr = curr.tb_next
        return exc_tb