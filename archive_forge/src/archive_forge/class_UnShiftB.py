from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
class UnShiftB(Operator):
    """ Increment a lower index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))
        self._ap = ap
        self._bq = bq
        self._i = i
        ap = list(ap)
        bq = list(bq)
        bi = bq.pop(i) + 1
        if bi == 0:
            raise ValueError('Cannot increment -1 lower index.')
        m = Poly(_x * (bi - 1), _x)
        for b in bq:
            m *= Poly(_x + b - 1, _x)
        B = Dummy('B')
        D = Poly((bi - 1) * B - bi + 1, B)
        n = Poly(z, B)
        for a in ap:
            n *= D + a.as_poly(B)
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment index: cancels with upper')
        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(B, _x / (bi - 1) + 1), _x)
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        return '<Increment lower index #%s of %s, %s.>' % (self._i, self._ap, self._bq)