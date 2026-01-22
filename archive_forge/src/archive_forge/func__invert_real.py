from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul, Equality,
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
from sympy.core.mod import Mod
from sympy.core.numbers import igcd, I, Number, Rational, oo, ilcm
from sympy.core.power import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
from sympy.calculus.util import periodicity, continuous_domain, function_range
from types import GeneratorType
def _invert_real(f, g_ys, symbol):
    """Helper function for _invert."""
    if f == symbol or g_ys is S.EmptySet:
        return (f, g_ys)
    n = Dummy('n', real=True)
    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        return _invert_real(f.exp, imageset(Lambda(n, log(n)), g_ys), symbol)
    if hasattr(f, 'inverse') and f.inverse() is not None and (not isinstance(f, (TrigonometricFunction, HyperbolicFunction))):
        if len(f.args) > 1:
            raise ValueError('Only functions with one argument are supported.')
        return _invert_real(f.args[0], imageset(Lambda(n, f.inverse()(n)), g_ys), symbol)
    if isinstance(f, Abs):
        return _invert_abs(f.args[0], g_ys, symbol)
    if f.is_Add:
        g, h = f.as_independent(symbol)
        if g is not S.Zero:
            return _invert_real(h, imageset(Lambda(n, n - g), g_ys), symbol)
    if f.is_Mul:
        g, h = f.as_independent(symbol)
        if g is not S.One:
            return _invert_real(h, imageset(Lambda(n, n / g), g_ys), symbol)
    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)
        if not expo_has_sym:
            if expo.is_rational:
                num, den = expo.as_numer_denom()
                if den % 2 == 0 and num % 2 == 1 and (den.is_zero is False):
                    root = Lambda(n, real_root(n, expo))
                    g_ys_pos = g_ys & Interval(0, oo)
                    res = imageset(root, g_ys_pos)
                    _inv, _set = _invert_real(base, res, symbol)
                    return (_inv, _set)
                if den % 2 == 1:
                    root = Lambda(n, real_root(n, expo))
                    res = imageset(root, g_ys)
                    if num % 2 == 0:
                        neg_res = imageset(Lambda(n, -n), res)
                        return _invert_real(base, res + neg_res, symbol)
                    if num % 2 == 1:
                        return _invert_real(base, res, symbol)
            elif expo.is_irrational:
                root = Lambda(n, real_root(n, expo))
                g_ys_pos = g_ys & Interval(0, oo)
                res = imageset(root, g_ys_pos)
                return _invert_real(base, res, symbol)
            else:
                pass
        if not base_has_sym:
            rhs = g_ys.args[0]
            if base.is_positive:
                return _invert_real(expo, imageset(Lambda(n, log(n, base, evaluate=False)), g_ys), symbol)
            elif base.is_negative:
                s, b = integer_log(rhs, base)
                if b:
                    return _invert_real(expo, FiniteSet(s), symbol)
                else:
                    return (expo, S.EmptySet)
            elif base.is_zero:
                one = Eq(rhs, 1)
                if one == S.true:
                    return _invert_real(expo, FiniteSet(0), symbol)
                elif one == S.false:
                    return (expo, S.EmptySet)
    if isinstance(f, TrigonometricFunction):
        if isinstance(g_ys, FiniteSet):

            def inv(trig):
                if isinstance(trig, (sin, csc)):
                    F = asin if isinstance(trig, sin) else acsc
                    return (lambda a: n * pi + S.NegativeOne ** n * F(a),)
                if isinstance(trig, (cos, sec)):
                    F = acos if isinstance(trig, cos) else asec
                    return (lambda a: 2 * n * pi + F(a), lambda a: 2 * n * pi - F(a))
                if isinstance(trig, (tan, cot)):
                    return (lambda a: n * pi + trig.inverse()(a),)
            n = Dummy('n', integer=True)
            invs = S.EmptySet
            for L in inv(f):
                invs += Union(*[imageset(Lambda(n, L(g)), S.Integers) for g in g_ys])
            return _invert_real(f.args[0], invs, symbol)
    return (f, g_ys)