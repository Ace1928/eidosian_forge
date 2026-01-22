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
class TrapezoidalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c', 'd')

    @property
    def set(self):
        return Interval(self.a, self.d)

    @staticmethod
    def check(a, b, c, d):
        _value_check(a < d, 'Lower bound parameter a < %s. a = %s' % (d, a))
        _value_check((a <= b, b < c), 'Level start parameter b must be in range [%s, %s). b = %s' % (a, c, b))
        _value_check((b < c, c <= d), 'Level end parameter c must be in range (%s, %s]. c = %s' % (b, d, c))
        _value_check(d >= c, 'Upper bound parameter d > %s. d = %s' % (c, d))

    def pdf(self, x):
        a, b, c, d = (self.a, self.b, self.c, self.d)
        return Piecewise((2 * (x - a) / ((b - a) * (d + c - a - b)), And(a <= x, x < b)), (2 / (d + c - a - b), And(b <= x, x < c)), (2 * (d - x) / ((d - c) * (d + c - a - b)), And(c <= x, x <= d)), (S.Zero, True))