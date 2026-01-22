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
def _compute_basis(self, closed_form):
    """
        Compute a set of functions B=(f1, ..., fn), a nxn matrix M
        and a 1xn matrix C such that:
           closed_form = C B
           z d/dz B = M B.
        """
    afactors = [_x + a for a in self.func.ap]
    bfactors = [_x + b - 1 for b in self.func.bq]
    expr = _x * Mul(*bfactors) - self.z * Mul(*afactors)
    poly = Poly(expr, _x)
    n = poly.degree() - 1
    b = [closed_form]
    for _ in range(n):
        b.append(self.z * b[-1].diff(self.z))
    self.B = Matrix(b)
    self.C = Matrix([[1] + [0] * n])
    m = eye(n)
    m = m.col_insert(0, zeros(n, 1))
    l = poly.all_coeffs()[1:]
    l.reverse()
    self.M = m.row_insert(n, -Matrix([l]) / poly.all_coeffs()[0])