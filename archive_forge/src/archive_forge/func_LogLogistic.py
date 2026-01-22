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
def LogLogistic(name, alpha, beta):
    """
    Create a continuous random variable with a log-logistic distribution.
    The distribution is unimodal when ``beta > 1``.

    Explanation
    ===========

    The density of the log-logistic distribution is given by

    .. math::
        f(x) := \\frac{(\\frac{\\beta}{\\alpha})(\\frac{x}{\\alpha})^{\\beta - 1}}
                {(1 + (\\frac{x}{\\alpha})^{\\beta})^2}

    Parameters
    ==========

    alpha : Real number, `\\alpha > 0`, scale parameter and median of distribution
    beta : Real number, `\\beta > 0`, a shape parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogLogistic, density, cdf, quantile
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> p = Symbol("p")
    >>> z = Symbol("z", positive=True)

    >>> X = LogLogistic("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                  beta - 1
           /  z  \\
      beta*|-----|
           \\alpha/
    ------------------------
                           2
          /       beta    \\
          |/  z  \\        |
    alpha*||-----|     + 1|
          \\\\alpha/        /

    >>> cdf(X)(z)
    1/(1 + (z/alpha)**(-beta))

    >>> quantile(X)(p)
    alpha*(p/(1 - p))**(1/beta)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Log-logistic_distribution

    """
    return rv(name, LogLogisticDistribution, (alpha, beta))