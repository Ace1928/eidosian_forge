import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.exceptions import ignore_warnings # noqa:F401
def _both_exp_pow(func):
    """
    Decorator used to run the test twice: the first time `e^x` is represented
    as ``Pow(E, x)``, the second time as ``exp(x)`` (exponential object is not
    a power).

    This is a temporary trick helping to manage the elimination of the class
    ``exp`` in favor of a replacement by ``Pow(E, ...)``.
    """
    from sympy.core.parameters import _exp_is_pow

    def func_wrap():
        with _exp_is_pow(True):
            func()
        with _exp_is_pow(False):
            func()
    wrapper = functools.update_wrapper(func_wrap, func)
    return wrapper