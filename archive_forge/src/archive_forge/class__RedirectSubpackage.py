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
class _RedirectSubpackage(ModuleType):
    """Redirect a subpackage to a subpackage.

    This allows all references like:

    >>> from numba.old_subpackage import module
    >>> module.item

    >>> import numba.old_subpackage.module
    >>> numba.old_subpackage.module.item

    >>> from numba.old_subpackage.module import item
    """

    def __init__(self, old_module_locals, new_module):
        old_module = old_module_locals['__name__']
        super().__init__(old_module)
        self.__old_module_states = {}
        self.__new_module = new_module
        new_mod_obj = import_module(new_module)
        for k, v in new_mod_obj.__dict__.items():
            setattr(self, k, v)
            if isinstance(v, ModuleType):
                sys.modules[f'{old_module}.{k}'] = sys.modules[v.__name__]
        for attr, value in old_module_locals.items():
            if attr.startswith('__') and attr.endswith('__'):
                if attr != '__builtins__':
                    setattr(self, attr, value)
                    self.__old_module_states[attr] = value

    def __reduce__(self):
        args = (self.__old_module_states, self.__new_module)
        return (_RedirectSubpackage, args)