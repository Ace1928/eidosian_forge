import operator
import typing as t
from collections import abc
from numbers import Number
from .runtime import Undefined
from .utils import pass_environment
def test_odd(value: int) -> bool:
    """Return true if the variable is odd."""
    return value % 2 == 1