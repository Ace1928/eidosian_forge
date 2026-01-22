import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def _basic_einsum_parse_input(operands):
    eq, *arrays = operands
    lhs, rhs = eq.split('->')
    return (lhs, rhs, arrays)