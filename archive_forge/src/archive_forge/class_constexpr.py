from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
class constexpr:
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:
        return f'constexpr[{self.value}]'

    def __index__(self):
        return self.value

    def __add__(self, other):
        return constexpr(self.value + other.value)

    def __radd__(self, other):
        return constexpr(other.value + self.value)

    def __sub__(self, other):
        return constexpr(self.value - other.value)

    def __rsub__(self, other):
        return constexpr(other.value - self.value)

    def __mul__(self, other):
        return constexpr(self.value * other.value)

    def __mod__(self, other):
        return constexpr(self.value % other.value)

    def __rmul__(self, other):
        return constexpr(other.value * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / other.value)

    def __rtruediv__(self, other):
        return constexpr(other.value / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // other.value)

    def __rfloordiv__(self, other):
        return constexpr(other.value // self.value)

    def __gt__(self, other):
        return constexpr(self.value > other.value)

    def __rgt__(self, other):
        return constexpr(other.value > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= other.value)

    def __rge__(self, other):
        return constexpr(other.value >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < other.value)

    def __rlt__(self, other):
        return constexpr(other.value < self.value)

    def __le__(self, other):
        return constexpr(self.value <= other.value)

    def __rle__(self, other):
        return constexpr(other.value <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == other.value)

    def __ne__(self, other):
        return constexpr(self.value != other.value)

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return constexpr(-self.value)

    def __and__(self, other):
        return constexpr(self.value & other.value)

    def logical_and(self, other):
        return constexpr(self.value and other.value)

    def __or__(self, other):
        return constexpr(self.value | other.value)

    def __xor__(self, other):
        return constexpr(self.value ^ other.value)

    def logical_or(self, other):
        return constexpr(self.value or other.value)

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value ** other.value)

    def __rshift__(self, other):
        return constexpr(self.value >> other.value)

    def __lshift__(self, other):
        return constexpr(self.value << other.value)

    def __not__(self):
        return constexpr(not self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)