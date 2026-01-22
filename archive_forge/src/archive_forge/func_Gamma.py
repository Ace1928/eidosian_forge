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
def Gamma(name, k, theta):
    """
    Create a continuous random variable with a Gamma distribution.

    Explanation
    ===========

    The density of the Gamma distribution is given by

    .. math::
        f(x) := \\frac{1}{\\Gamma(k) \\theta^k} x^{k - 1} e^{-\\frac{x}{\\theta}}

    with :math:`x \\in [0,1]`.

    Parameters
    ==========

    k : Real number, `k > 0`, a shape
    theta : Real number, `\\theta > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gamma, density, cdf, E, variance
    >>> from sympy import Symbol, pprint, simplify

    >>> k = Symbol("k", positive=True)
    >>> theta = Symbol("theta", positive=True)
    >>> z = Symbol("z")

    >>> X = Gamma("x", k, theta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                      -z
                    -----
         -k  k - 1  theta
    theta  *z     *e
    ---------------------
           Gamma(k)

    >>> C = cdf(X, meijerg=True)(z)
    >>> pprint(C, use_unicode=False)
    /            /     z  \\
    |k*lowergamma|k, -----|
    |            \\   theta/
    <----------------------  for z >= 0
    |     Gamma(k + 1)
    |
    \\          0             otherwise

    >>> E(X)
    k*theta

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
           2
    k*theta


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_distribution
    .. [2] https://mathworld.wolfram.com/GammaDistribution.html

    """
    return rv(name, GammaDistribution, (k, theta))