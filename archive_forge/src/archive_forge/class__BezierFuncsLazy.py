from fontTools.pens.basePen import BasePen
from functools import partial
from itertools import count
import sympy as sp
import sys
class _BezierFuncsLazy(dict):

    def __init__(self, symfunc):
        self._symfunc = symfunc
        self._bezfuncs = {}

    def __missing__(self, i):
        args = ['p%d' % d for d in range(i + 1)]
        f = green(self._symfunc, BezierCurve[i])
        f = sp.gcd_terms(f.collect(sum(P, ())))
        return sp.lambdify(args, f)