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
def Logistic(name, mu, s):
    """
    Create a continuous random variable with a logistic distribution.

    Explanation
    ===========

    The density of the logistic distribution is given by

    .. math::
        f(x) := \\frac{e^{-(x-\\mu)/s}} {s\\left(1+e^{-(x-\\mu)/s}\\right)^2}

    Parameters
    ==========

    mu : Real number, the location (mean)
    s : Real number, `s > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logistic, density, cdf
    >>> from sympy import Symbol

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")

    >>> X = Logistic("x", mu, s)

    >>> density(X)(z)
    exp((mu - z)/s)/(s*(exp((mu - z)/s) + 1)**2)

    >>> cdf(X)(z)
    1/(exp((mu - z)/s) + 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logistic_distribution
    .. [2] https://mathworld.wolfram.com/LogisticDistribution.html

    """
    return rv(name, LogisticDistribution, (mu, s))