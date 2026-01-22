from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def _interpolate_axis(self, axis, v):
    i = self._find_interval(v)
    v = rinterpolate(self.intervals[i - 1], self.intervals[i], v)
    return interpolate(self.colors[i - 1][axis], self.colors[i][axis], v)