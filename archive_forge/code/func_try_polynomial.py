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
def try_polynomial(func, z):
    """ Recognise polynomial cases. Returns None if not such a case.
        Requires order to be fully reduced. """
    abuckets, bbuckets = (sift(func.ap, _mod1), sift(func.bq, _mod1))
    a0 = abuckets[S.Zero]
    b0 = bbuckets[S.Zero]
    a0.sort()
    b0.sort()
    al0 = [x for x in a0 if x <= 0]
    bl0 = [x for x in b0 if x <= 0]
    if bl0 and all((a < bl0[-1] for a in al0)):
        return oo
    if not al0:
        return None
    a = al0[-1]
    fac = 1
    res = S.One
    for n in Tuple(*list(range(-a))):
        fac *= z
        fac /= n + 1
        fac *= Mul(*[a + n for a in func.ap])
        fac /= Mul(*[b + n for b in func.bq])
        res += fac
    return res