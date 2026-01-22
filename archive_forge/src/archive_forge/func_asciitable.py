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
def asciitable(columns, rows):
    """Formats an ascii table for given columns and rows.

    Parameters
    ----------
    columns : list
        The column names
    rows : list of tuples
        The rows in the table. Each tuple must be the same length as
        ``columns``.
    """
    rows = [tuple((str(i) for i in r)) for r in rows]
    columns = tuple((str(i) for i in columns))
    widths = tuple((max(max(map(len, x)), len(c)) for x, c in zip(zip(*rows), columns)))
    row_template = ('|' + ' %%-%ds |' * len(columns)) % widths
    header = row_template % tuple(columns)
    bar = '+%s+' % '+'.join(('-' * (w + 2) for w in widths))
    data = '\n'.join((row_template % r for r in rows))
    return '\n'.join([bar, header, bar, data, bar])