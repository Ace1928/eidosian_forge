import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
class orthopoly1d(np.poly1d):

    def __init__(self, roots, weights=None, hn=1.0, kn=1.0, wfunc=None, limits=None, monic=False, eval_func=None):
        equiv_weights = [weights[k] / wfunc(roots[k]) for k in range(len(roots))]
        mu = sqrt(hn)
        if monic:
            evf = eval_func
            if evf:
                knn = kn

                def eval_func(x):
                    return evf(x) / knn
            mu = mu / abs(kn)
            kn = 1.0
        poly = np.poly1d(roots, r=True)
        np.poly1d.__init__(self, poly.coeffs * float(kn))
        self.weights = np.array(list(zip(roots, weights, equiv_weights)))
        self.weight_func = wfunc
        self.limits = limits
        self.normcoef = mu
        self._eval_func = eval_func

    def __call__(self, v):
        if self._eval_func and (not isinstance(v, np.poly1d)):
            return self._eval_func(v)
        else:
            return np.poly1d.__call__(self, v)

    def _scale(self, p):
        if p == 1.0:
            return
        self._coeffs *= p
        evf = self._eval_func
        if evf:
            self._eval_func = lambda x: evf(x) * p
        self.normcoef *= p