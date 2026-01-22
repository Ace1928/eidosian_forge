from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
def _parametrize_ternary_quadratic(solution, _var, coeff):
    assert 1 not in coeff
    x_0, y_0, z_0 = solution
    v = list(_var)
    if x_0 is None:
        return (None, None, None)
    if solution.count(0) >= 2:
        return (None, None, None)
    if x_0 == 0:
        v[0], v[1] = (v[1], v[0])
        y_p, x_p, z_p = _parametrize_ternary_quadratic((y_0, x_0, z_0), v, coeff)
        return (x_p, y_p, z_p)
    x, y, z = v
    r, p, q = symbols('r, p, q', integer=True)
    eq = sum((k * v for k, v in coeff.items()))
    eq_1 = _mexpand(eq.subs(zip((x, y, z), (r * x_0, r * y_0 + p, r * z_0 + q))))
    A, B = eq_1.as_independent(r, as_Add=True)
    x = A * x_0
    y = A * y_0 - _mexpand(B / r * p)
    z = A * z_0 - _mexpand(B / r * q)
    return _remove_gcd(x, y, z)