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
class NthAlgebraic(SingleODESolver):
    """
    Solves an `n`\\th order ordinary differential equation using algebra and
    integrals.

    There is no general form for the kind of equation that this can solve. The
    the equation is solved algebraically treating differentiation as an
    invertible algebraic function.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(f(x) * (f(x).diff(x)**2 - 1), 0)
    >>> dsolve(eq, f(x), hint='nth_algebraic')
    [Eq(f(x), 0), Eq(f(x), C1 - x), Eq(f(x), C1 + x)]

    Note that this solver can return algebraic solutions that do not have any
    integration constants (f(x) = 0 in the above example).
    """
    hint = 'nth_algebraic'
    has_integral = True

    def _matches(self):
        """
        Matches any differential equation that nth_algebraic can solve. Uses
        `sympy.solve` but teaches it how to integrate derivatives.

        This involves calling `sympy.solve` and does most of the work of finding a
        solution (apart from evaluating the integrals).
        """
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        var = self.ode_problem.sym
        diffx = self._get_diffx(var)

        def replace(eq, var):

            def expand_diffx(*args):
                differand, diffs = (args[0], args[1:])
                toreplace = differand
                for v, n in diffs:
                    for _ in range(n):
                        if v == var:
                            toreplace = diffx(toreplace)
                        else:
                            toreplace = Derivative(toreplace, v)
                return toreplace
            return eq.replace(Derivative, expand_diffx)

        def unreplace(eq, var):
            return eq.replace(diffx, lambda e: Derivative(e, var))
        subs_eqn = replace(eq, var)
        try:
            solns = solve(subs_eqn, func, simplify=False)
        except NotImplementedError:
            solns = []
        solns = [simplify(unreplace(soln, var)) for soln in solns]
        solns = [Equality(func, soln) for soln in solns]
        self.solutions = solns
        return len(solns) != 0

    def _get_general_solution(self, *, simplify_flag: bool=True):
        return self.solutions
    _diffx_stored: dict[Symbol, type[Function]] = {}

    @staticmethod
    def _get_diffx(var):
        diffcls = NthAlgebraic._diffx_stored.get(var, None)
        if diffcls is None:

            class diffx(Function):

                def inverse(self):
                    return lambda expr: Integral(expr, var) + Dummy('C')
            diffcls = NthAlgebraic._diffx_stored.setdefault(var, diffx)
        return diffcls