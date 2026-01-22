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
def _warn_old_style():
    from numba.core import errors
    exccls, _, tb = sys.exc_info()
    numba_errs = (errors.NumbaError, errors.NumbaWarning)
    if exccls is not None and (not issubclass(exccls, numba_errs)):
        tb_last = traceback.format_tb(tb)[-1]
        msg = f'{_old_style_deprecation_msg}\nException origin:\n{tb_last}'
        warnings.warn(msg, errors.NumbaPendingDeprecationWarning)