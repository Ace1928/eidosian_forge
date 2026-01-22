from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions import assoc_legendre
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
class Znm(Function):
    """
    Real spherical harmonics defined as

    .. math::

        Z_n^m(\\theta, \\varphi) :=
        \\begin{cases}
          \\frac{Y_n^m(\\theta, \\varphi) + \\overline{Y_n^m(\\theta, \\varphi)}}{\\sqrt{2}} &\\quad m > 0 \\\\
          Y_n^m(\\theta, \\varphi) &\\quad m = 0 \\\\
          \\frac{Y_n^m(\\theta, \\varphi) - \\overline{Y_n^m(\\theta, \\varphi)}}{i \\sqrt{2}} &\\quad m < 0 \\\\
        \\end{cases}

    which gives in simplified form

    .. math::

        Z_n^m(\\theta, \\varphi) =
        \\begin{cases}
          \\frac{Y_n^m(\\theta, \\varphi) + (-1)^m Y_n^{-m}(\\theta, \\varphi)}{\\sqrt{2}} &\\quad m > 0 \\\\
          Y_n^m(\\theta, \\varphi) &\\quad m = 0 \\\\
          \\frac{Y_n^m(\\theta, \\varphi) - (-1)^m Y_n^{-m}(\\theta, \\varphi)}{i \\sqrt{2}} &\\quad m < 0 \\\\
        \\end{cases}

    Examples
    ========

    >>> from sympy import Znm, Symbol, simplify
    >>> from sympy.abc import n, m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")
    >>> Znm(n, m, theta, phi)
    Znm(n, m, theta, phi)

    For specific integers n and m we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Znm(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))
    >>> simplify(Znm(1, 1, theta, phi).expand(func=True))
    -sqrt(3)*sin(theta)*cos(phi)/(2*sqrt(pi))
    >>> simplify(Znm(2, 1, theta, phi).expand(func=True))
    -sqrt(15)*sin(2*theta)*cos(phi)/(4*sqrt(pi))

    See Also
    ========

    Ynm, Ynm_c

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/

    """

    @classmethod
    def eval(cls, n, m, theta, phi):
        if m.is_positive:
            zz = (Ynm(n, m, theta, phi) + Ynm_c(n, m, theta, phi)) / sqrt(2)
            return zz
        elif m.is_zero:
            return Ynm(n, m, theta, phi)
        elif m.is_negative:
            zz = (Ynm(n, m, theta, phi) - Ynm_c(n, m, theta, phi)) / (sqrt(2) * I)
            return zz