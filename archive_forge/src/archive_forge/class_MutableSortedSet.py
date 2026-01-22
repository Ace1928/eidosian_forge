import atexit
import builtins
import functools
import inspect
import os
import operator
import timeit
import math
import sys
import traceback
import weakref
import warnings
import threading
import contextlib
import typing as _tp
from types import ModuleType
from importlib import import_module
import numpy as np
from inspect import signature as pysignature # noqa: F401
from inspect import Signature as pySignature # noqa: F401
from inspect import Parameter as pyParameter # noqa: F401
from numba.core.config import (PYVERSION, MACHINE_BITS, # noqa: F401
from numba.core import config
from numba.core import types
from collections.abc import Mapping, Sequence, MutableSet, MutableMapping
class MutableSortedSet(MutableSet[T], _tp.Generic[T]):
    """Mutable Sorted Set
    """

    def __init__(self, values: _tp.Iterable[T]=()):
        self._values = set(values)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter((k for k in sorted(self._values)))

    def __contains__(self, x: T) -> bool:
        return self._values.__contains__(x)

    def add(self, x: T):
        return self._values.add(x)

    def discard(self, value: T):
        self._values.discard(value)

    def update(self, values):
        self._values.update(values)