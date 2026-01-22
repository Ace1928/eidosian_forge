from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
class Secant:
    """
    1d-solver generating pairs of approximative root and error.

    Needs starting points x0 and x1 close to the root.
    x1 defaults to x0 + 0.25.

    Pro:

    * converges fast

    Contra:

    * converges slowly for multiple roots
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) == 1:
            self.x0 = x0[0]
            self.x1 = self.x0 + 0.25
        elif len(x0) == 2:
            self.x0 = x0[0]
            self.x1 = x0[1]
        else:
            raise ValueError('expected 1 or 2 starting points, got %i' % len(x0))
        self.f = f

    def __iter__(self):
        f = self.f
        x0 = self.x0
        x1 = self.x1
        f0 = f(x0)
        while True:
            f1 = f(x1)
            l = x1 - x0
            if not l:
                break
            s = (f1 - f0) / l
            if not s:
                break
            x0, x1 = (x1, x1 - f1 / s)
            f0 = f1
            yield (x1, abs(l))