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
def ChiNoncentral(name, k, l):
    """
    Create a continuous random variable with a non-central Chi distribution.

    Explanation
    ===========

    The density of the non-central Chi distribution is given by

    .. math::
        f(x) := \\frac{e^{-(x^2+\\lambda^2)/2} x^k\\lambda}
                {(\\lambda x)^{k/2}} I_{k/2-1}(\\lambda x)

    with `x \\geq 0`. Here, `I_\\nu (x)` is the
    :ref:`modified Bessel function of the first kind <besseli>`.

    Parameters
    ==========

    k : A positive Integer, $k > 0$
        The number of degrees of freedom.
    lambda : Real number, `\\lambda > 0`
        Shift parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiNoncentral, density
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True)
    >>> l = Symbol("l")
    >>> z = Symbol("z")

    >>> X = ChiNoncentral("x", k, l)

    >>> density(X)(z)
    l*z**k*exp(-l**2/2 - z**2/2)*besseli(k/2 - 1, l*z)/(l*z)**(k/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Noncentral_chi_distribution
    """
    return rv(name, ChiNoncentralDistribution, (k, l))