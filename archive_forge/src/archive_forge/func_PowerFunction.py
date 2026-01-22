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
def PowerFunction(name, alpha, a, b):
    """
    Creates a continuous random variable with a Power Function Distribution.

    Explanation
    ===========

    The density of PowerFunction distribution is given by

    .. math::
        f(x) := \\frac{{\\alpha}(x - a)^{\\alpha - 1}}{(b - a)^{\\alpha}}

    with :math:`x \\in [a,b]`.

    Parameters
    ==========

    alpha : Positive number, `0 < \\alpha`, the shape parameter
    a : Real number, :math:`-\\infty < a`, the left boundary
    b : Real number, :math:`a < b < \\infty`, the right boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import PowerFunction, density, cdf, E, variance
    >>> from sympy import Symbol
    >>> alpha = Symbol("alpha", positive=True)
    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = PowerFunction("X", 2, a, b)

    >>> density(X)(z)
    (-2*a + 2*z)/(-a + b)**2

    >>> cdf(X)(z)
    Piecewise((a**2/(a**2 - 2*a*b + b**2) - 2*a*z/(a**2 - 2*a*b + b**2) +
    z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))

    >>> alpha = 2
    >>> a = 0
    >>> b = 1
    >>> Y = PowerFunction("Y", alpha, a, b)

    >>> E(Y)
    2/3

    >>> variance(Y)
    1/18

    References
    ==========

    .. [1] https://web.archive.org/web/20200204081320/http://www.mathwave.com/help/easyfit/html/analyses/distributions/power_func.html

    """
    return rv(name, PowerFunctionDistribution, (alpha, a, b))