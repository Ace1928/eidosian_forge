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
def Erlang(name, k, l):
    """
    Create a continuous random variable with an Erlang distribution.

    Explanation
    ===========

    The density of the Erlang distribution is given by

    .. math::
        f(x) := \\frac{\\lambda^k x^{k-1} e^{-\\lambda x}}{(k-1)!}

    with :math:`x \\in [0,\\infty]`.

    Parameters
    ==========

    k : Positive integer
    l : Real number, `\\lambda > 0`, the rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Erlang, density, cdf, E, variance
    >>> from sympy import Symbol, simplify, pprint

    >>> k = Symbol("k", integer=True, positive=True)
    >>> l = Symbol("l", positive=True)
    >>> z = Symbol("z")

    >>> X = Erlang("x", k, l)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     k  k - 1  -l*z
    l *z     *e
    ---------------
        Gamma(k)

    >>> C = cdf(X)(z)
    >>> pprint(C, use_unicode=False)
    /lowergamma(k, l*z)
    |------------------  for z > 0
    <     Gamma(k)
    |
    \\        0           otherwise


    >>> E(X)
    k/l

    >>> simplify(variance(X))
    k/l**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Erlang_distribution
    .. [2] https://mathworld.wolfram.com/ErlangDistribution.html

    """
    return rv(name, GammaDistribution, (k, S.One / l))