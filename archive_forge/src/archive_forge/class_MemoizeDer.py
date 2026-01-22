import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
class MemoizeDer:
    """Decorator that caches the value and derivative(s) of function each
    time it is called.

    This is a simplistic memoizer that calls and caches a single value
    of `f(x, *args)`.
    It assumes that `args` does not change between invocations.
    It supports the use case of a root-finder where `args` is fixed,
    `x` changes, and only rarely, if at all, does x assume the same value
    more than once."""

    def __init__(self, fun):
        self.fun = fun
        self.vals = None
        self.x = None
        self.n_calls = 0

    def __call__(self, x, *args):
        """Calculate f or use cached value if available"""
        if self.vals is None or x != self.x:
            fg = self.fun(x, *args)
            self.x = x
            self.n_calls += 1
            self.vals = fg[:]
        return self.vals[0]

    def fprime(self, x, *args):
        """Calculate f' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[1]

    def fprime2(self, x, *args):
        """Calculate f'' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[2]

    def ncalls(self):
        return self.n_calls