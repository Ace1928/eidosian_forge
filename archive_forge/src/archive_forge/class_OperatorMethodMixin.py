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
class OperatorMethodMixin:
    """A mixin for dynamically implementing operators"""
    __slots__ = ()

    @classmethod
    def _bind_operator(cls, op):
        """bind operator to this class"""
        name = op.__name__
        if name.endswith('_'):
            name = name[:-1]
        elif name == 'inv':
            name = 'invert'
        meth = f'__{name}__'
        if name in ('abs', 'invert', 'neg', 'pos'):
            setattr(cls, meth, cls._get_unary_operator(op))
        else:
            setattr(cls, meth, cls._get_binary_operator(op))
            if name in ('eq', 'gt', 'ge', 'lt', 'le', 'ne', 'getitem'):
                return
            rmeth = f'__r{name}__'
            setattr(cls, rmeth, cls._get_binary_operator(op, inv=True))

    @classmethod
    def _get_unary_operator(cls, op):
        """Must return a method used by unary operator"""
        raise NotImplementedError

    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        """Must return a method used by binary operator"""
        raise NotImplementedError