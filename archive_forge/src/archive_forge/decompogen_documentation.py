from sympy.core import (Function, Pow, sympify, Expr)
from sympy.core.relational import Relational
from sympy.core.singleton import S
from sympy.polys import Poly, decompose
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Min, Max

    Returns the composition of functions.
    Given a list of functions ``g_s``, returns their composition ``f``,
    where:
        f = g_1 o g_2 o .. o g_n

    Note: This is a General composition function. It also composes Polynomials.
    For only Polynomial composition see ``compose`` in polys.

    Examples
    ========

    >>> from sympy.solvers.decompogen import compogen
    >>> from sympy.abc import x
    >>> from sympy import sqrt, sin, cos
    >>> compogen([sin(x), cos(x)], x)
    sin(cos(x))
    >>> compogen([x**2 + x + 1, sin(x)], x)
    sin(x)**2 + sin(x) + 1
    >>> compogen([sqrt(x), 6*x**2 - 5], x)
    sqrt(6*x**2 - 5)
    >>> compogen([sin(x), sqrt(x), cos(x), x**2 + 1], x)
    sin(sqrt(cos(x**2 + 1)))
    >>> compogen([x**2 - x - 1, x**2 + x], x)
    -x**2 - x + (x**2 + x)**2 - 1
    