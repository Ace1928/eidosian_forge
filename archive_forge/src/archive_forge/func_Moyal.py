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
def Moyal(name, mu, sigma):
    """
    Create a continuous random variable with a Moyal distribution.

    Explanation
    ===========

    The density of the Moyal distribution is given by

    .. math::
        f(x) := \\frac{\\exp-\\frac{1}{2}\\exp-\\frac{x-\\mu}{\\sigma}-\\frac{x-\\mu}{2\\sigma}}{\\sqrt{2\\pi}\\sigma}

    with :math:`x \\in \\mathbb{R}`.

    Parameters
    ==========

    mu : Real number
        Location parameter
    sigma : Real positive number
        Scale parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Moyal, density, cdf
    >>> from sympy import Symbol, simplify
    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True, real=True)
    >>> z = Symbol("z")
    >>> X = Moyal("x", mu, sigma)
    >>> density(X)(z)
    sqrt(2)*exp(-exp((mu - z)/sigma)/2 - (-mu + z)/(2*sigma))/(2*sqrt(pi)*sigma)
    >>> simplify(cdf(X)(z))
    1 - erf(sqrt(2)*exp((mu - z)/(2*sigma))/2)

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/MoyalDistribution.html
    .. [2] https://www.stat.rice.edu/~dobelman/textfiles/DistributionsHandbook.pdf

    """
    return rv(name, MoyalDistribution, (mu, sigma))