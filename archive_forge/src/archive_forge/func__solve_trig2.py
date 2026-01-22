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
def _solve_trig2(f, symbol, domain):
    """Secondary helper to solve trigonometric equations,
    called when first helper fails """
    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(sin, cos, tan, sec, cot, csc)
    trig_arguments = [e.args[0] for e in trig_functions]
    denominators = []
    numerators = []
    if not trig_functions:
        return ConditionSet(symbol, Eq(f_original, 0), domain)
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise ValueError('give up, we cannot solve if this is not a polynomial in x')
        if poly_ar.degree() > 1:
            raise ValueError('degree of variable inside polynomial should not exceed one')
        if poly_ar.degree() == 0:
            continue
        c = poly_ar.all_coeffs()[0]
        try:
            numerators.append(Rational(c).p)
            denominators.append(Rational(c).q)
        except TypeError:
            return ConditionSet(symbol, Eq(f_original, 0), domain)
    x = Dummy('x')
    if len(numerators) > 1:
        mu = Rational(2) * ilcm(*denominators) / igcd(*numerators)
    else:
        assert len(numerators) == 1
        mu = Rational(2) * denominators[0] / numerators[0]
    f = f.subs(symbol, mu * x)
    f = f.rewrite(tan)
    f = expand_trig(f)
    f = together(f)
    g, h = fraction(f)
    y = Dummy('y')
    g, h = (g.expand(), h.expand())
    g, h = (g.subs(tan(x), y), h.subs(tan(x), y))
    if g.has(x) or h.has(x):
        return ConditionSet(symbol, Eq(f_original, 0), domain)
    solns = solveset(g, y, S.Reals) - solveset(h, y, S.Reals)
    if isinstance(solns, FiniteSet):
        result = Union(*[invert_real(tan(symbol / mu), s, symbol)[1] for s in solns])
        dsol = invert_real(tan(symbol / mu), oo, symbol)[1]
        if degree(h) > degree(g):
            result = Union(result, dsol)
        return Intersection(result, domain)
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        return ConditionSet(symbol, Eq(f_original, 0), S.Reals)