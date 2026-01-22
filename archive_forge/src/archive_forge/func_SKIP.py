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
def SKIP(reason):
    """Similar to ``skip()``, but this is a decorator. """

    def wrapper(func):

        def func_wrapper():
            raise Skipped(reason)
        func_wrapper = functools.update_wrapper(func_wrapper, func)
        return func_wrapper
    return wrapper