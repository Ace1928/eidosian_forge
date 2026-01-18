from collections import defaultdict
from sympy.concrete import product
from sympy.core.singleton import S
from sympy.core.numbers import Rational, I
from sympy.core.symbol import Symbol, Wild, Dummy
from sympy.core.relational import Equality
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.simplify import simplify, hypersimp, hypersimilar  # type: ignore
from sympy.solvers import solve, solve_undetermined_coeffs
from sympy.polys import Poly, quo, gcd, lcm, roots, resultant
from sympy.functions import binomial, factorial, FallingFactorial, RisingFactorial
from sympy.matrices import Matrix, casoratian
from sympy.utilities.iterables import numbered_symbols

    Solve univariate recurrence with rational coefficients.

    Given `k`-th order linear recurrence `\operatorname{L} y = f`,
    or equivalently:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              \cdots + a_{0}(n) y(n) = f(n)

    where `a_{i}(n)`, for `i=0, \ldots, k`, are polynomials or rational
    functions in `n`, and `f` is a hypergeometric function or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms in
    `n`, finds all solutions or returns ``None``, if none were found.

    Initial conditions can be given as a dictionary in two forms:

        (1) ``{  n_0  : v_0,   n_1  : v_1, ...,   n_m  : v_m}``
        (2) ``{y(n_0) : v_0, y(n_1) : v_1, ..., y(n_m) : v_m}``

    or as a list ``L`` of values:

        ``L = [v_0, v_1, ..., v_m]``

    where ``L[i] = v_i``, for `i=0, \ldots, m`, maps to `y(n_i)`.

    Examples
    ========

    Lets consider the following recurrence:

    .. math:: (n - 1) y(n + 2) - (n^2 + 3 n - 2) y(n + 1) +
              2 n (n + 1) y(n) = 0

    >>> from sympy import Function, rsolve
    >>> from sympy.abc import n
    >>> y = Function('y')

    >>> f = (n - 1)*y(n + 2) - (n**2 + 3*n - 2)*y(n + 1) + 2*n*(n + 1)*y(n)

    >>> rsolve(f, y(n))
    2**n*C0 + C1*factorial(n)

    >>> rsolve(f, y(n), {y(0):0, y(1):3})
    3*2**n - 3*factorial(n)

    See Also
    ========

    rsolve_poly, rsolve_ratio, rsolve_hyper

    