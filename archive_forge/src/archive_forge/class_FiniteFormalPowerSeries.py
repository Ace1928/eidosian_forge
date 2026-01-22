from collections import defaultdict
from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable
class FiniteFormalPowerSeries(FormalPowerSeries):
    """Base Class for Product, Compose and Inverse classes"""

    def __init__(self, *args):
        pass

    @property
    def ffps(self):
        return self.args[0]

    @property
    def gfps(self):
        return self.args[1]

    @property
    def f(self):
        return self.ffps.function

    @property
    def g(self):
        return self.gfps.function

    @property
    def infinite(self):
        raise NotImplementedError('No infinite version for an object of FiniteFormalPowerSeries class.')

    def _eval_terms(self, n):
        raise NotImplementedError('(%s)._eval_terms()' % self)

    def _eval_term(self, pt):
        raise NotImplementedError('By the current logic, one can get termsupto a certain order, instead of getting term by term.')

    def polynomial(self, n):
        return self._eval_terms(n)

    def truncate(self, n=6):
        ffps = self.ffps
        pt_xk = ffps.xk.coeff(n)
        x, x0 = (ffps.x, ffps.x0)
        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def _eval_derivative(self, x):
        raise NotImplementedError

    def integrate(self, x):
        raise NotImplementedError