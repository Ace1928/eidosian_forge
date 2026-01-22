import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
@staticmethod
def _wrap_fget(orig_fget) -> Callable[[Any], Any]:
    if isinstance(orig_fget, classmethod):
        orig_fget = orig_fget.__func__

    @functools.wraps(orig_fget)
    def fget(obj):
        return orig_fget(obj.__class__)
    return fget