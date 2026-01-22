from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random
def Frechet(name, a, s=1, m=0):
    """
    Create a continuous random variable with a Frechet distribution.

    Explanation
    ===========

    The density of the Frechet distribution is given by

    .. math::
        f(x) := \\frac{\\alpha}{s} \\left(\\frac{x-m}{s}\\right)^{-1-\\alpha}
                 e^{-(\\frac{x-m}{s})^{-\\alpha}}

    with :math:`x \\geq m`.

    Parameters
    ==========

    a : Real number, :math:`a \\in \\left(0, \\infty\\right)` the shape
    s : Real number, :math:`s \\in \\left(0, \\infty\\right)` the scale
    m : Real number, :math:`m \\in \\left(-\\infty, \\infty\\right)` the minimum

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Frechet, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", positive=True)
    >>> s = Symbol("s", positive=True)
    >>> m = Symbol("m", real=True)
    >>> z = Symbol("z")

    >>> X = Frechet("x", a, s, m)

    >>> density(X)(z)
    a*((-m + z)/s)**(-a - 1)*exp(-1/((-m + z)/s)**a)/s

    >>> cdf(X)(z)
    Piecewise((exp(-1/((-m + z)/s)**a), m <= z), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution

    """
    return rv(name, FrechetDistribution, (a, s, m))