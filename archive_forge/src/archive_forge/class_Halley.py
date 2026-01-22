from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
class Halley:
    """
    1d-solver generating pairs of approximative root and error.

    Needs a starting point x0 close to the root.
    Uses Halley's method with cubic convergence rate.

    Pro:

    * converges even faster the Newton's method
    * useful when computing with *many* digits

    Contra:

    * needs first and second derivative of f
    * 3 function evaluations per iteration
    * converges slowly for multiple roots
    """
    maxsteps = 20

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if not len(x0) == 1:
            raise ValueError('expected 1 starting point, got %i' % len(x0))
        self.x0 = x0[0]
        self.f = f
        if not 'df' in kwargs:

            def df(x):
                return self.ctx.diff(f, x)
        else:
            df = kwargs['df']
        self.df = df
        if not 'd2f' in kwargs:

            def d2f(x):
                return self.ctx.diff(df, x)
        else:
            d2f = kwargs['df']
        self.d2f = d2f

    def __iter__(self):
        x = self.x0
        f = self.f
        df = self.df
        d2f = self.d2f
        while True:
            prevx = x
            fx = f(x)
            dfx = df(x)
            d2fx = d2f(x)
            x -= 2 * fx * dfx / (2 * dfx ** 2 - fx * d2fx)
            error = abs(x - prevx)
            yield (x, error)