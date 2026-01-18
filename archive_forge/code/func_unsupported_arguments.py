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
def unsupported_arguments(doc, args):
    """Mark unsupported arguments with a disclaimer"""
    lines = doc.split('\n')
    for arg in args:
        subset = [(i, line) for i, line in enumerate(lines) if re.match('^\\s*' + arg + ' ?:', line)]
        if len(subset) == 1:
            [(i, line)] = subset
            lines[i] = line + '  (Not supported in Dask)'
    return '\n'.join(lines)