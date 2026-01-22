from __future__ import annotations
from typing import ClassVar, Iterator
from .riccati import match_riccati, solve_riccati
from sympy.core import Add, S, Pow, Rational
from sympy.core.cache import cached_property
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, Function, expand, Subs, _mexpand
from sympy.core.numbers import zoo
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Dummy, Wild
from sympy.core.mul import Mul
from sympy.functions import exp, tan, log, sqrt, besselj, bessely, cbrt, airyai, airybi
from sympy.integrals import Integral
from sympy.polys import Poly
from sympy.polys.polytools import cancel, factor, degree
from sympy.simplify import collect, simplify, separatevars, logcombine, posify # type: ignore
from sympy.simplify.radsimp import fraction
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.solvers.deutils import ode_order, _preprocess
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.solvers import PolyNonlinearError
from .hypergeometric import equivalence_hypergeometric, match_2nd_2F1_hypergeometric, \
from .nonhomogeneous import _get_euler_characteristic_eq_sols, _get_const_characteristic_eq_sols, \
from .lie_group import _ode_lie_group
from .ode import dsolve, ode_sol_simplicity, odesimp, homogeneous_order
class AlmostLinear(SinglePatternODESolver):
    """
    Solves an almost-linear differential equation.

    The general form of an almost linear differential equation is

    .. math:: a(x) g'(f(x)) f'(x) + b(x) g(f(x)) + c(x)

    Here `f(x)` is the function to be solved for (the dependent variable).
    The substitution `g(f(x)) = u(x)` leads to a linear differential equation
    for `u(x)` of the form `a(x) u' + b(x) u + c(x) = 0`. This can be solved
    for `u(x)` by the `first_linear` hint and then `f(x)` is found by solving
    `g(f(x)) = u(x)`.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.FirstLinear`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint, sin, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = x*d + x*f(x) + 1
    >>> dsolve(eq, f(x), hint='almost_linear')
    Eq(f(x), (C1 - Ei(x))*exp(-x))
    >>> pprint(dsolve(eq, f(x), hint='almost_linear'))
                        -x
    f(x) = (C1 - Ei(x))*e
    >>> example = cos(f(x))*f(x).diff(x) + sin(f(x)) + 1
    >>> pprint(example)
                        d
    sin(f(x)) + cos(f(x))*--(f(x)) + 1
                        dx
    >>> pprint(dsolve(example, f(x), hint='almost_linear'))
                    /    -x    \\             /    -x    \\
    [f(x) = pi - asin\\C1*e   - 1/, f(x) = asin\\C1*e   - 1/]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = 'almost_linear'
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return (P, Q)

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return P * fx.diff(x) + Q

    def _verify(self, fx):
        a, b = self.wilds_match()
        c, b = b.as_independent(fx) if b.is_Add else (S.Zero, b)
        if b.diff(fx) != 0 and (not simplify(b.diff(fx) / a).has(fx)):
            self.ly = factor_terms(b).as_independent(fx, as_Add=False)[1]
            self.ax = a / self.ly.diff(fx)
            self.cx = -c
            self.bx = factor_terms(b) / self.ly
            return True
        return False

    def _get_general_solution(self, *, simplify_flag: bool=True):
        x = self.ode_problem.sym
        C1, = self.ode_problem.get_numbered_constants(num=1)
        gensol = Eq(self.ly, (C1 + Integral(self.cx / self.ax * exp(Integral(self.bx / self.ax, x)), x)) * exp(-Integral(self.bx / self.ax, x)))
        return [gensol]