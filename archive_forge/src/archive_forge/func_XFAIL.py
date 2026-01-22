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
def XFAIL(func):

    def wrapper():
        try:
            func()
        except Exception as e:
            message = str(e)
            if message != 'Timeout':
                raise XFail(func.__name__)
            else:
                raise Skipped('Timeout')
        raise XPass(func.__name__)
    wrapper = functools.update_wrapper(wrapper, func)
    return wrapper