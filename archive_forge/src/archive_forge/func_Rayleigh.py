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
def Rayleigh(name, sigma):
    """
    Create a continuous random variable with a Rayleigh distribution.

    Explanation
    ===========

    The density of the Rayleigh distribution is given by

    .. math ::
        f(x) := \\frac{x}{\\sigma^2} e^{-x^2/2\\sigma^2}

    with :math:`x > 0`.

    Parameters
    ==========

    sigma : Real number, `\\sigma > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Rayleigh, density, E, variance
    >>> from sympy import Symbol

    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = Rayleigh("x", sigma)

    >>> density(X)(z)
    z*exp(-z**2/(2*sigma**2))/sigma**2

    >>> E(X)
    sqrt(2)*sqrt(pi)*sigma/2

    >>> variance(X)
    -pi*sigma**2/2 + 2*sigma**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rayleigh_distribution
    .. [2] https://mathworld.wolfram.com/RayleighDistribution.html

    """
    return rv(name, RayleighDistribution, (sigma,))