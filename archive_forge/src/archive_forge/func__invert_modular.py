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
def _invert_modular(modterm, rhs, n, symbol):
    """
    Helper function to invert modular equation.
    ``Mod(a, m) - rhs = 0``

    Generally it is inverted as (a, ImageSet(Lambda(n, m*n + rhs), S.Integers)).
    More simplified form will be returned if possible.

    If it is not invertible then (modterm, rhs) is returned.

    The following cases arise while inverting equation ``Mod(a, m) - rhs = 0``:

    1. If a is symbol then  m*n + rhs is the required solution.

    2. If a is an instance of ``Add`` then we try to find two symbol independent
       parts of a and the symbol independent part gets transferred to the other
       side and again the ``_invert_modular`` is called on the symbol
       dependent part.

    3. If a is an instance of ``Mul`` then same as we done in ``Add`` we separate
       out the symbol dependent and symbol independent parts and transfer the
       symbol independent part to the rhs with the help of invert and again the
       ``_invert_modular`` is called on the symbol dependent part.

    4. If a is an instance of ``Pow`` then two cases arise as following:

        - If a is of type (symbol_indep)**(symbol_dep) then the remainder is
          evaluated with the help of discrete_log function and then the least
          period is being found out with the help of totient function.
          period*n + remainder is the required solution in this case.
          For reference: (https://en.wikipedia.org/wiki/Euler's_theorem)

        - If a is of type (symbol_dep)**(symbol_indep) then we try to find all
          primitive solutions list with the help of nthroot_mod function.
          m*n + rem is the general solution where rem belongs to solutions list
          from nthroot_mod function.

    Parameters
    ==========

    modterm, rhs : Expr
        The modular equation to be inverted, ``modterm - rhs = 0``

    symbol : Symbol
        The variable in the equation to be inverted.

    n : Dummy
        Dummy variable for output g_n.

    Returns
    =======

    A tuple (f_x, g_n) is being returned where f_x is modular independent function
    of symbol and g_n being set of values f_x can have.

    Examples
    ========

    >>> from sympy import symbols, exp, Mod, Dummy, S
    >>> from sympy.solvers.solveset import _invert_modular as invert_modular
    >>> x, y = symbols('x y')
    >>> n = Dummy('n')
    >>> invert_modular(Mod(exp(x), 7), S(5), n, x)
    (Mod(exp(x), 7), 5)
    >>> invert_modular(Mod(x, 7), S(5), n, x)
    (x, ImageSet(Lambda(_n, 7*_n + 5), Integers))
    >>> invert_modular(Mod(3*x + 8, 7), S(5), n, x)
    (x, ImageSet(Lambda(_n, 7*_n + 6), Integers))
    >>> invert_modular(Mod(x**4, 7), S(5), n, x)
    (x, EmptySet)
    >>> invert_modular(Mod(2**(x**2 + x + 1), 7), S(2), n, x)
    (x**2 + x + 1, ImageSet(Lambda(_n, 3*_n + 1), Naturals0))

    """
    a, m = modterm.args
    if rhs.is_real is False or any((term.is_real is False for term in list(_term_factors(a)))):
        return (modterm, rhs)
    if abs(rhs) >= abs(m):
        return (symbol, S.EmptySet)
    if a == symbol:
        return (symbol, ImageSet(Lambda(n, m * n + rhs), S.Integers))
    if a.is_Add:
        g, h = a.as_independent(symbol)
        if g is not S.Zero:
            x_indep_term = rhs - Mod(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)
    if a.is_Mul:
        g, h = a.as_independent(symbol)
        if g is not S.One:
            x_indep_term = rhs * invert(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)
    if a.is_Pow:
        base, expo = a.args
        if expo.has(symbol) and (not base.has(symbol)):
            try:
                remainder = discrete_log(m / igcd(m, rhs), rhs, a.base)
            except ValueError:
                return (modterm, rhs)
            period = totient(m)
            for p in divisors(period):
                if pow(a.base, p, m / igcd(m, a.base)) == 1:
                    period = p
                    break
            return (expo, ImageSet(Lambda(n, period * n + remainder), S.Naturals0))
        elif base.has(symbol) and (not expo.has(symbol)):
            try:
                remainder_list = nthroot_mod(rhs, expo, m, all_roots=True)
                if remainder_list == []:
                    return (symbol, S.EmptySet)
            except (ValueError, NotImplementedError):
                return (modterm, rhs)
            g_n = S.EmptySet
            for rem in remainder_list:
                g_n += ImageSet(Lambda(n, m * n + rem), S.Integers)
            return (base, g_n)
    return (modterm, rhs)