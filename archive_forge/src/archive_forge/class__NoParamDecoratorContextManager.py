import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast
class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses."""

    def __new__(cls, orig_func=None):
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)