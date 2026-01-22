import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def _new_duck_type_and_implements():
    """Create a duck array type and implements functions."""
    HANDLED_FUNCTIONS = {}

    class MyArray:

        def __array_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all((issubclass(t, MyArray) for t in types)):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(numpy_function):
        """Register an __array_function__ implementations."""

        def decorator(func):
            HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator
    return (MyArray, implements)