import contextlib
import contextvars
import dataclasses
import functools
import importlib
import inspect
import os
import re
import sys
import traceback
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, overload, Set, Tuple, Type, TypeVar
import numpy as np
import pandas as pd
import sympy
import sympy.printing.repr
from cirq._doc import document
def _is_internal(filename: str) -> bool:
    """Returns whether filename is internal to python.

    This is similar to how the built-in warnings module differentiates frames from internal modules.
    It is specific to CPython - see
    https://github.com/python/cpython/blob/41ec17e45d54473d32f543396293256f1581e44d/Lib/warnings.py#L275.
    """
    return 'importlib' in filename and '_bootstrap' in filename