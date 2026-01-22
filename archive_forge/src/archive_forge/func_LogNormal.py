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
def LogNormal(name, mean, std):
    """
    Create a continuous random variable with a log-normal distribution.

    Explanation
    ===========

    The density of the log-normal distribution is given by

    .. math::
        f(x) := \\frac{1}{x\\sqrt{2\\pi\\sigma^2}}
                e^{-\\frac{\\left(\\ln x-\\mu\\right)^2}{2\\sigma^2}}

    with :math:`x \\geq 0`.

    Parameters
    ==========

    mu : Real number
        The log-scale.
    sigma : Real number
        A shape. ($\\sigma^2 > 0$)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogNormal, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = LogNormal("x", mu, sigma)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                          2
           -(-mu + log(z))
           -----------------
                      2
      ___      2*sigma
    \\/ 2 *e
    ------------------------
            ____
        2*\\/ pi *sigma*z


    >>> X = LogNormal('x', 0, 1) # Mean 0, standard deviation 1

    >>> density(X)(z)
    sqrt(2)*exp(-log(z)**2/2)/(2*sqrt(pi)*z)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lognormal
    .. [2] https://mathworld.wolfram.com/LogNormalDistribution.html

    """
    return rv(name, LogNormalDistribution, (mean, std))