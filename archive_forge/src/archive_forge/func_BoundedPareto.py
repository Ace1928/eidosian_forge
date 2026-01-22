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
def BoundedPareto(name, alpha, left, right):
    """
    Create a continuous random variable with a Bounded Pareto distribution.

    The density of the Bounded Pareto distribution is given by

    .. math::
        f(x) := \\frac{\\alpha L^{\\alpha}x^{-\\alpha-1}}{1-(\\frac{L}{H})^{\\alpha}}

    Parameters
    ==========

    alpha : Real Number, `\\alpha > 0`
        Shape parameter
    left : Real Number, `left > 0`
        Location parameter
    right : Real Number, `right > left`
        Location parameter

    Examples
    ========

    >>> from sympy.stats import BoundedPareto, density, cdf, E
    >>> from sympy import symbols
    >>> L, H = symbols('L, H', positive=True)
    >>> X = BoundedPareto('X', 2, L, H)
    >>> x = symbols('x')
    >>> density(X)(x)
    2*L**2/(x**3*(1 - L**2/H**2))
    >>> cdf(X)(x)
    Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) + H**2/(H**2 - L**2), L <= x), (0, True))
    >>> E(X).simplify()
    2*H*L/(H + L)

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution

    """
    return rv(name, BoundedParetoDistribution, (alpha, left, right))