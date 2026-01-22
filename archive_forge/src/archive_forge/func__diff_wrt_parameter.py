from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
def _diff_wrt_parameter(self, idx):
    an = list(self.an)
    ap = list(self.aother)
    bm = list(self.bm)
    bq = list(self.bother)
    if idx < len(an):
        an.pop(idx)
    else:
        idx -= len(an)
        if idx < len(ap):
            ap.pop(idx)
        else:
            idx -= len(ap)
            if idx < len(bm):
                bm.pop(idx)
            else:
                bq.pop(idx - len(bm))
    pairs1 = []
    pairs2 = []
    for l1, l2, pairs in [(an, bq, pairs1), (ap, bm, pairs2)]:
        while l1:
            x = l1.pop()
            found = None
            for i, y in enumerate(l2):
                if not Mod((x - y).simplify(), 1):
                    found = i
                    break
            if found is None:
                raise NotImplementedError('Derivative not expressible as G-function?')
            y = l2[i]
            l2.pop(i)
            pairs.append((x, y))
    res = log(self.argument) * self
    for a, b in pairs1:
        sign = 1
        n = a - b
        base = b
        if n < 0:
            sign = -1
            n = b - a
            base = a
        for k in range(n):
            res -= sign * meijerg(self.an + (base + k + 1,), self.aother, self.bm, self.bother + (base + k + 0,), self.argument)
    for a, b in pairs2:
        sign = 1
        n = b - a
        base = a
        if n < 0:
            sign = -1
            n = a - b
            base = b
        for k in range(n):
            res -= sign * meijerg(self.an, self.aother + (base + k + 1,), self.bm + (base + k + 0,), self.bother, self.argument)
    return res