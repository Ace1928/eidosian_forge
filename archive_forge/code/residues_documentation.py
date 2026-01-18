from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.utilities.timeutils import timethis

    Finds the residue of ``expr`` at the point x=x0.

    The residue is defined as the coefficient of ``1/(x-x0)`` in the power series
    expansion about ``x=x0``.

    Examples
    ========

    >>> from sympy import Symbol, residue, sin
    >>> x = Symbol("x")
    >>> residue(1/x, x, 0)
    1
    >>> residue(1/x**2, x, 0)
    0
    >>> residue(2/sin(x), x, 0)
    2

    This function is essential for the Residue Theorem [1].

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Residue_theorem
    