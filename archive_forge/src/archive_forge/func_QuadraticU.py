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
def QuadraticU(name, a, b):
    """
    Create a Continuous Random Variable with a U-quadratic distribution.

    Explanation
    ===========

    The density of the U-quadratic distribution is given by

    .. math::
        f(x) := \\alpha (x-\\beta)^2

    with :math:`x \\in [a,b]`.

    Parameters
    ==========

    a : Real number
    b : Real number, :math:`a < b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import QuadraticU, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = QuadraticU("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /                2
    |   /  a   b    \\
    |12*|- - - - + z|
    |   \\  2   2    /
    <-----------------  for And(b >= z, a <= z)
    |            3
    |    (-a + b)
    |
    \\        0                 otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/U-quadratic_distribution

    """
    return rv(name, QuadraticUDistribution, (a, b))