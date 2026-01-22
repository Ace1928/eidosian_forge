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
def FDistribution(name, d1, d2):
    """
    Create a continuous random variable with a F distribution.

    Explanation
    ===========

    The density of the F distribution is given by

    .. math::
        f(x) := \\frac{\\sqrt{\\frac{(d_1 x)^{d_1} d_2^{d_2}}
                {(d_1 x + d_2)^{d_1 + d_2}}}}
                {x \\mathrm{B} \\left(\\frac{d_1}{2}, \\frac{d_2}{2}\\right)}

    with :math:`x > 0`.

    Parameters
    ==========

    d1 : `d_1 > 0`, where `d_1` is the degrees of freedom (`n_1 - 1`)
    d2 : `d_2 > 0`, where `d_2` is the degrees of freedom (`n_2 - 1`)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import FDistribution, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FDistribution("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
      d2
      --    ______________________________
      2    /       d1            -d1 - d2
    d2  *\\/  (d1*z)  *(d1*z + d2)
    --------------------------------------
                    /d1  d2\\
                 z*B|--, --|
                    \\2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/F-distribution
    .. [2] https://mathworld.wolfram.com/F-Distribution.html

    """
    return rv(name, FDistributionDistribution, (d1, d2))