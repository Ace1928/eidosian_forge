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
def GammaInverse(name, a, b):
    """
    Create a continuous random variable with an inverse Gamma distribution.

    Explanation
    ===========

    The density of the inverse Gamma distribution is given by

    .. math::
        f(x) := \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} x^{-\\alpha - 1}
                \\exp\\left(\\frac{-\\beta}{x}\\right)

    with :math:`x > 0`.

    Parameters
    ==========

    a : Real number, `a > 0`, a shape
    b : Real number, `b > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import GammaInverse, density, cdf
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = GammaInverse("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                -b
                ---
     a  -a - 1   z
    b *z      *e
    ---------------
       Gamma(a)

    >>> cdf(X)(z)
    Piecewise((uppergamma(a, b/z)/gamma(a), z > 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse-gamma_distribution

    """
    return rv(name, GammaInverseDistribution, (a, b))