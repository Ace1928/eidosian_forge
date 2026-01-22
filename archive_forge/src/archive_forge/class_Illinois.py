from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
class Illinois:
    """
    1d-solver generating pairs of approximative root and error.

    Uses Illinois method or similar to find a root of f in [a, b].
    Might fail for multiple roots (needs sign change).
    Combines bisect with secant (improved regula falsi).

    The only difference between the methods is the scaling factor m, which is
    used to ensure convergence (you can choose one using the 'method' keyword):

    Illinois method ('illinois'):
        m = 0.5

    Pegasus method ('pegasus'):
        m = fb/(fb + fz)

    Anderson-Bjoerk method ('anderson'):
        m = 1 - fz/fb if positive else 0.5

    Pro:

    * converges very fast

    Contra:

    * has problems with multiple roots
    * needs sign change
    """
    maxsteps = 30

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        if len(x0) != 2:
            raise ValueError('expected interval of 2 points, got %i' % len(x0))
        self.a = x0[0]
        self.b = x0[1]
        self.f = f
        self.tol = kwargs['tol']
        self.verbose = kwargs['verbose']
        self.method = kwargs.get('method', 'illinois')
        self.getm = _getm(self.method)
        if self.verbose:
            print('using %s method' % self.method)

    def __iter__(self):
        method = self.method
        f = self.f
        a = self.a
        b = self.b
        fa = f(a)
        fb = f(b)
        m = None
        while True:
            l = b - a
            if l == 0:
                break
            s = (fb - fa) / l
            z = a - fa / s
            fz = f(z)
            if abs(fz) < self.tol:
                if self.verbose:
                    print('canceled with z =', z)
                yield (z, l)
                break
            if fz * fb < 0:
                a = b
                fa = fb
                b = z
                fb = fz
            else:
                m = self.getm(fz, fb)
                b = z
                fb = fz
                fa = m * fa
            if self.verbose and m and (not method == 'illinois'):
                print('m:', m)
            yield ((a + b) / 2, abs(l))