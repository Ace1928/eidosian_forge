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
class MeijerUnShiftA(Operator):
    """ Decrement an upper b index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))
        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i
        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bm.pop(i) - 1
        m = Poly(1, _x) * prod((Poly(b - _x, _x) for b in bm)) * prod((Poly(_x - b, _x) for b in bq))
        A = Dummy('A')
        D = Poly(bi - A, A)
        n = Poly(z, A) * prod((D + 1 - a for a in an)) * prod((-D + a - 1 for a in ap))
        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper b index (cancels)')
        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, bi - _x), _x)
        self._poly = Poly((m - n) / b0, _x)

    def __str__(self):
        return '<Decrement upper b index #%s of %s, %s, %s, %s.>' % (self._i, self._an, self._ap, self._bm, self._bq)