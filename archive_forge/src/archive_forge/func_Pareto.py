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
def Pareto(name, xm, alpha):
    """
    Create a continuous random variable with the Pareto distribution.

    Explanation
    ===========

    The density of the Pareto distribution is given by

    .. math::
        f(x) := \\frac{\\alpha\\,x_m^\\alpha}{x^{\\alpha+1}}

    with :math:`x \\in [x_m,\\infty]`.

    Parameters
    ==========

    xm : Real number, `x_m > 0`, a scale
    alpha : Real number, `\\alpha > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Pareto, density
    >>> from sympy import Symbol

    >>> xm = Symbol("xm", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = Pareto("x", xm, beta)

    >>> density(X)(z)
    beta*xm**beta*z**(-beta - 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution
    .. [2] https://mathworld.wolfram.com/ParetoDistribution.html

    """
    return rv(name, ParetoDistribution, (xm, alpha))