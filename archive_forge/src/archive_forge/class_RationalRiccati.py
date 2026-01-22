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
class RationalRiccati(SinglePatternODESolver):
    """
    Gives general solutions to the first order Riccati differential
    equations that have atleast one rational particular solution.

    .. math :: y' = b_0(x) + b_1(x) y + b_2(x) y^2

    where `b_0`, `b_1` and `b_2` are rational functions of `x`
    with `b_2 \\ne 0` (`b_2 = 0` would make it a Bernoulli equation).

    Examples
    ========

    >>> from sympy import Symbol, Function, dsolve, checkodesol
    >>> f = Function('f')
    >>> x = Symbol('x')

    >>> eq = -x**4*f(x)**2 + x**3*f(x).diff(x) + x**2*f(x) + 20
    >>> sol = dsolve(eq, hint="1st_rational_riccati")
    >>> sol
    Eq(f(x), (4*C1 - 5*x**9 - 4)/(x**2*(C1 + x**9 - 1)))
    >>> checkodesol(eq, sol)
    (True, 0)

    References
    ==========

    - Riccati ODE:  https://en.wikipedia.org/wiki/Riccati_equation
    - N. Thieu Vo - Rational and Algebraic Solutions of First-Order Algebraic ODEs:
      Algorithm 11, pp. 78 - https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf
    """
    has_integral = False
    hint = '1st_rational_riccati'
    order = [1]

    def _wilds(self, f, x, order):
        b0 = Wild('b0', exclude=[f(x), f(x).diff(x)])
        b1 = Wild('b1', exclude=[f(x), f(x).diff(x)])
        b2 = Wild('b2', exclude=[f(x), f(x).diff(x)])
        return (b0, b1, b2)

    def _equation(self, fx, x, order):
        b0, b1, b2 = self.wilds()
        return fx.diff(x) - b0 - b1 * fx - b2 * fx ** 2

    def _matches(self):
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        if order != 1:
            return False
        match, funcs = match_riccati(eq, f, x)
        if not match:
            return False
        _b0, _b1, _b2 = funcs
        b0, b1, b2 = self.wilds()
        self._wilds_match = match = {b0: _b0, b1: _b1, b2: _b2}
        return True

    def _get_general_solution(self, *, simplify_flag: bool=True):
        b0, b1, b2 = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return solve_riccati(fx, x, b0, b1, b2, gensol=True)