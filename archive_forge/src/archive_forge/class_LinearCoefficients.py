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
class LinearCoefficients(HomogeneousCoeffBest):
    """
    Solves a differential equation with linear coefficients.

    The general form of a differential equation with linear coefficients is

    .. math:: y' + F\\left(\\!\\frac{a_1 x + b_1 y + c_1}{a_2 x + b_2 y +
                c_2}\\!\\right) = 0\\text{,}

    where `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are constants and `a_1 b_2
    - a_2 b_1 \\ne 0`.

    This can be solved by substituting:

    .. math:: x = x' + \\frac{b_2 c_1 - b_1 c_2}{a_2 b_1 - a_1 b_2}

              y = y' + \\frac{a_1 c_2 - a_2 c_1}{a_2 b_1 - a_1
                  b_2}\\text{.}

    This substitution reduces the equation to a homogeneous differential
    equation.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffBest`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> df = f(x).diff(x)
    >>> eq = (x + f(x) + 1)*df + (f(x) - 6*x + 1)
    >>> dsolve(eq, hint='linear_coefficients')
    [Eq(f(x), -x - sqrt(C1 + 7*x**2) - 1), Eq(f(x), -x + sqrt(C1 + 7*x**2) - 1)]
    >>> pprint(dsolve(eq, hint='linear_coefficients'))
                      ___________                     ___________
                   /         2                     /         2
    [f(x) = -x - \\/  C1 + 7*x   - 1, f(x) = -x + \\/  C1 + 7*x   - 1]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = 'linear_coefficients'
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return (d, e)

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e * fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        a, b = self.wilds()
        F = self.d / self.e
        x = self.ode_problem.sym
        params = self._linear_coeff_match(F, fx)
        if params:
            self.xarg, self.yarg = params
            u = Dummy('u')
            t = Dummy('t')
            self.y = Dummy('y')
            dummy_eq = self.ode_problem.eq.subs(((fx.diff(x), t), (fx, u)))
            reps = ((x, x + self.xarg), (u, u + self.yarg), (t, fx.diff(x)), (u, fx))
            dummy_eq = simplify(dummy_eq.subs(reps))
            r2 = collect(expand(dummy_eq), [fx.diff(x), fx]).match(a * fx.diff(x) + b)
            if r2:
                self.d, self.e = (r2[b], r2[a])
                orderd = homogeneous_order(self.d, x, fx)
                ordere = homogeneous_order(self.e, x, fx)
                if orderd == ordere and orderd is not None:
                    self.d = self.d.subs(fx, self.y)
                    self.e = self.e.subs(fx, self.y)
                    return True
                return False
            return False

    def _linear_coeff_match(self, expr, func):
        """
        Helper function to match hint ``linear_coefficients``.

        Matches the expression to the form `(a_1 x + b_1 f(x) + c_1)/(a_2 x + b_2
        f(x) + c_2)` where the following conditions hold:

        1. `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are Rationals;
        2. `c_1` or `c_2` are not equal to zero;
        3. `a_2 b_1 - a_1 b_2` is not equal to zero.

        Return ``xarg``, ``yarg`` where

        1. ``xarg`` = `(b_2 c_1 - b_1 c_2)/(a_2 b_1 - a_1 b_2)`
        2. ``yarg`` = `(a_1 c_2 - a_2 c_1)/(a_2 b_1 - a_1 b_2)`


        Examples
        ========

        >>> from sympy import Function, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import LinearCoefficients
        >>> f = Function('f')
        >>> eq = (-25*f(x) - 8*x + 62)/(4*f(x) + 11*x - 11)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (1/9, 22/9)
        >>> eq = sin((-5*f(x) - 8*x + 6)/(4*f(x) + x - 1))
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (19/27, 2/27)
        >>> eq = sin(f(x)/x)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))

        """
        f = func.func
        x = func.args[0]

        def abc(eq):
            """
            Internal function of _linear_coeff_match
            that returns Rationals a, b, c
            if eq is a*x + b*f(x) + c, else None.
            """
            eq = _mexpand(eq)
            c = eq.as_independent(x, f(x), as_Add=True)[0]
            if not c.is_Rational:
                return
            a = eq.coeff(x)
            if not a.is_Rational:
                return
            b = eq.coeff(f(x))
            if not b.is_Rational:
                return
            if eq == a * x + b * f(x) + c:
                return (a, b, c)

        def match(arg):
            """
            Internal function of _linear_coeff_match that returns Rationals a1,
            b1, c1, a2, b2, c2 and a2*b1 - a1*b2 of the expression (a1*x + b1*f(x)
            + c1)/(a2*x + b2*f(x) + c2) if one of c1 or c2 and a2*b1 - a1*b2 is
            non-zero, else None.
            """
            n, d = arg.together().as_numer_denom()
            m = abc(n)
            if m is not None:
                a1, b1, c1 = m
                m = abc(d)
                if m is not None:
                    a2, b2, c2 = m
                    d = a2 * b1 - a1 * b2
                    if (c1 or c2) and d:
                        return (a1, b1, c1, a2, b2, c2, d)
        m = [fi.args[0] for fi in expr.atoms(Function) if fi.func != f and len(fi.args) == 1 and (not fi.args[0].is_Function)] or {expr}
        m1 = match(m.pop())
        if m1 and all((match(mi) == m1 for mi in m)):
            a1, b1, c1, a2, b2, c2, denom = m1
            return ((b2 * c1 - b1 * c2) / denom, (a1 * c2 - a2 * c1) / denom)

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        u = Dummy('u')
        return [self.d, self.e, fx, x, u, self.u1, self.y, self.xarg, self.yarg]