import itertools
from sympy.calculus.util import (continuous_domain, periodicity,
from sympy.core import Symbol, Dummy, sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Relational, Eq, Ge, Lt
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.core.singleton import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.complexes import im, Abs
from sympy.logic import And
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr
from sympy.polys.polyutils import _nsort
from sympy.solvers.solveset import solvify, solveset
from sympy.utilities.iterables import sift, iterable
from sympy.utilities.misc import filldedent
Reduce a system of inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import reduce_inequalities

    >>> reduce_inequalities(0 <= x + 3, [])
    (-3 <= x) & (x < oo)

    >>> reduce_inequalities(0 <= x + y*2 - 1, [x])
    (x < oo) & (x >= 1 - 2*y)
    