from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import (Integer, Rational)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Or
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.stats.frv import (SingleFiniteDistribution,
from sympy.stats.rv import _value_check, Density, is_random
from sympy.utilities.iterables import multiset
from sympy.utilities.misc import filldedent
class RobustSolitonDistribution(SingleFiniteDistribution):
    _argnames = ('k', 'delta', 'c')

    @staticmethod
    def check(k, delta, c):
        _value_check(k.is_integer and k.is_positive, "'k' must be a positive integer")
        _value_check(Gt(delta, 0) and Le(delta, 1), "'delta' must be a real number in the interval (0,1)")
        _value_check(c.is_positive, "'c' must be a positive real number.")

    @property
    def R(self):
        return self.c * log(self.k / self.delta) * self.k ** 0.5

    @property
    def Z(self):
        z = 0
        for i in Range(1, round(self.k / self.R)):
            z += 1 / i
        z += log(self.R / self.delta)
        return 1 + z * self.R / self.k

    @property
    def low(self):
        return S.One

    @property
    def high(self):
        return self.k

    @property
    def set(self):
        return set(map(Integer, range(1, self.k + 1)))

    @property
    def is_symbolic(self):
        return not (self.k.is_number and self.c.is_number and self.delta.is_number)

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        rho = Piecewise((Rational(1, self.k), cond1), (Rational(1, x * (x - 1)), cond2), (S.Zero, True))
        cond1 = Ge(x, 1) & Le(x, round(self.k / self.R) - 1)
        cond2 = Eq(x, round(self.k / self.R))
        tau = Piecewise((self.R / (self.k * x), cond1), (self.R * log(self.R / self.delta) / self.k, cond2), (S.Zero, True))
        return (rho + tau) / self.Z