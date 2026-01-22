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
def Laplace(name, mu, b):
    """
    Create a continuous random variable with a Laplace distribution.

    Explanation
    ===========

    The density of the Laplace distribution is given by

    .. math::
        f(x) := \\frac{1}{2 b} \\exp \\left(-\\frac{|x-\\mu|}b \\right)

    Parameters
    ==========

    mu : Real number or a list/matrix, the location (mean) or the
        location vector
    b : Real number or a positive definite matrix, representing a scale
        or the covariance matrix.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Laplace, density, cdf
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu")
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Laplace("x", mu, b)

    >>> density(X)(z)
    exp(-Abs(mu - z)/b)/(2*b)

    >>> cdf(X)(z)
    Piecewise((exp((-mu + z)/b)/2, mu > z), (1 - exp((mu - z)/b)/2, True))

    >>> L = Laplace('L', [1, 2], [[1, 0], [0, 1]])
    >>> pprint(density(L)(1, 2), use_unicode=False)
     5        /     ____\\
    e *besselk\\0, \\/ 35 /
    ---------------------
              pi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laplace_distribution
    .. [2] https://mathworld.wolfram.com/LaplaceDistribution.html

    """
    if isinstance(mu, (list, MatrixBase)) and isinstance(b, (list, MatrixBase)):
        from sympy.stats.joint_rv_types import MultivariateLaplace
        return MultivariateLaplace(name, mu, b)
    return rv(name, LaplaceDistribution, (mu, b))