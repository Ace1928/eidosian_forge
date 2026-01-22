from __future__ import annotations
import itertools
from contextlib import contextmanager
from itertools import chain
from threading import local
from typing import Any, Callable, TYPE_CHECKING, Union
from unittest.mock import patch
import sympy
from torch._inductor.utils import IndentedBuffer
from torch.fx.graph import inplace_methods, magic_methods
from .utils import reduction_num_outputs, sympy_str, sympy_symbol
class OpsWrapper:
    """This wraps any returned IR values into an `OpsValue` instance, so that we
    can overload the magic methods for writing mathematical expressions fluently.
    """

    def __getattr__(self, name):

        def inner(*args, **kwargs):
            new_args = [OpsWrapper._unwrap(a) for a in args]
            new_kwargs = {k: OpsWrapper._unwrap(v) for k, v in kwargs.items()}
            return OpsWrapper._wrap(getattr(_ops, name)(*new_args, **new_kwargs))
        return inner

    @staticmethod
    def _unwrap(x):
        if isinstance(x, (list, tuple)):
            return tuple((OpsWrapper._unwrap(v) for v in x))
        if isinstance(x, OpsValue):
            return x.value
        return x

    @staticmethod
    def _wrap(x):
        if isinstance(x, (list, tuple)):
            return tuple((OpsValue(v) for v in x))
        return OpsValue(x)

    @staticmethod
    def indirect_indexing(index, size, check=True):
        index = OpsWrapper._unwrap(index)
        return _ops.indirect_indexing(index, size, check)