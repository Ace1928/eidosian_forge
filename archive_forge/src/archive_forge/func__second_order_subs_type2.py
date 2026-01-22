from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate
def _second_order_subs_type2(A, funcs, t_):
    """
    Returns a second order system based on the coefficient matrix passed.

    Explanation
    ===========

    This function returns a system of second order ODE of the following form:

    .. math::
        X'' = A * X

    Here, $X$ is the vector of dependent variables, but a bit modified, $A$ is the
    coefficient matrix passed.

    Along with returning the second order system, this function also returns the new
    dependent variables with the new independent variable `t_` passed.

    Parameters
    ==========

    A: Matrix
        Coefficient matrix of the system
    funcs: List
        List of old dependent variables
    t_: Symbol
        New independent variable

    Returns
    =======

    List, List

    """
    func_names = [func.func.__name__ for func in funcs]
    new_funcs = [Function(Dummy('{}_'.format(name)))(t_) for name in func_names]
    rhss = A * Matrix(new_funcs)
    new_eqs = [Eq(func.diff(t_, 2), rhs) for func, rhs in zip(new_funcs, rhss)]
    return (new_eqs, new_funcs)