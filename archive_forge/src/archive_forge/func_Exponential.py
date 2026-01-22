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
def Exponential(name, rate):
    """
    Create a continuous random variable with an Exponential distribution.

    Explanation
    ===========

    The density of the exponential distribution is given by

    .. math::
        f(x) := \\lambda \\exp(-\\lambda x)

    with $x > 0$. Note that the expected value is `1/\\lambda`.

    Parameters
    ==========

    rate : A positive Real number, `\\lambda > 0`, the rate (or inverse scale/inverse mean)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Exponential, density, cdf, E
    >>> from sympy.stats import variance, std, skewness, quantile
    >>> from sympy import Symbol

    >>> l = Symbol("lambda", positive=True)
    >>> z = Symbol("z")
    >>> p = Symbol("p")
    >>> X = Exponential("x", l)

    >>> density(X)(z)
    lambda*exp(-lambda*z)

    >>> cdf(X)(z)
    Piecewise((1 - exp(-lambda*z), z >= 0), (0, True))

    >>> quantile(X)(p)
    -log(1 - p)/lambda

    >>> E(X)
    1/lambda

    >>> variance(X)
    lambda**(-2)

    >>> skewness(X)
    2

    >>> X = Exponential('x', 10)

    >>> density(X)(z)
    10*exp(-10*z)

    >>> E(X)
    1/10

    >>> std(X)
    1/10

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponential_distribution
    .. [2] https://mathworld.wolfram.com/ExponentialDistribution.html

    """
    return rv(name, ExponentialDistribution, (rate,))