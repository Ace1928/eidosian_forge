import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
@classmethod
@contextmanager
def wrapattr(cls, stream, method, total=None, bytes=True, **tqdm_kwargs):
    """
        stream  : file-like object.
        method  : str, "read" or "write". The result of `read()` and
            the first argument of `write()` should have a `len()`.

        >>> with tqdm.wrapattr(file_obj, "read", total=file_obj.size) as fobj:
        ...     while True:
        ...         chunk = fobj.read(chunk_size)
        ...         if not chunk:
        ...             break
        """
    with cls(total=total, **tqdm_kwargs) as t:
        if bytes:
            t.unit = 'B'
            t.unit_scale = True
            t.unit_divisor = 1024
        yield CallbackIOWrapper(t.update, stream, method)