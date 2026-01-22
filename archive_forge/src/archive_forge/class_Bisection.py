from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
class Bisection:
    """
    1d-solver generating pairs of approximative root and error.

    Uses bisection method to find a root of f in [a, b].
    Might fail for multiple roots (needs sign change).

    Pro:

    * robust and reliable

    Contra:

    * converges slowly
    * needs sign change
    """
    maxsteps = 100

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.f = f
        self.a = x0[0]
        self.b = x0[1]

    def __iter__(self):
        f = self.f
        a = self.a
        b = self.b
        l = b - a
        fb = f(b)
        while True:
            m = self.ctx.ldexp(a + b, -1)
            fm = f(m)
            sign = fm * fb
            if sign < 0:
                a = m
            elif sign > 0:
                b = m
                fb = fm
            else:
                yield (m, self.ctx.zero)
            l /= 2
            yield ((a + b) / 2, abs(l))