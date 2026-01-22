from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (I, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Intersection
from sympy.core.containers import Dict
from sympy.core.logic import Logic
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet
from sympy.stats.rv import (RandomDomain, ProductDomain, ConditionalDomain,
class SingleFinitePSpace(SinglePSpace, FinitePSpace):
    """
    A single finite probability space

    Represents the probabilities of a set of random events that can be
    attributed to a single variable/symbol.

    This class is implemented by many of the standard FiniteRV types such as
    Die, Bernoulli, Coin, etc....
    """

    @property
    def domain(self):
        return SingleFiniteDomain(self.symbol, self.distribution.set)

    @property
    def _is_symbolic(self):
        """
        Helper property to check if the distribution
        of the random variable is having symbolic
        dimension.
        """
        return self.distribution.is_symbolic

    @property
    def distribution(self):
        return self.args[1]

    def pmf(self, expr):
        return self.distribution.pmf(expr)

    @property
    @cacheit
    def _density(self):
        return {FiniteSet((self.symbol, val)): prob for val, prob in self.distribution.dict.items()}

    @cacheit
    def compute_characteristic_function(self, expr):
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            return Lambda(t, Sum(d(ki) * exp(I * ki * t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_characteristic_function(expr)

    @cacheit
    def compute_moment_generating_function(self, expr):
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            return Lambda(t, Sum(d(ki) * exp(ki * t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_moment_generating_function(expr)

    def compute_quantile(self, expr):
        if self._is_symbolic:
            raise NotImplementedError('Computing quantile for random variables with symbolic dimension because the bounds of searching the required value is undetermined.')
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_quantile(expr)

    def compute_density(self, expr):
        if self._is_symbolic:
            rv = list(random_symbols(expr))[0]
            k = Dummy('k', integer=True)
            cond = True if not isinstance(expr, (Relational, Logic)) else expr.subs(rv, k)
            return Lambda(k, Piecewise((self.pmf(k), And(k >= self.args[1].low, k <= self.args[1].high, cond)), (S.Zero, True)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_density(expr)

    def compute_cdf(self, expr):
        if self._is_symbolic:
            d = self.compute_density(expr)
            k = Dummy('k')
            ki = Dummy('ki')
            return Lambda(k, Sum(d(ki), (ki, self.args[1].low, k)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_cdf(expr)

    def compute_expectation(self, expr, rvs=None, **kwargs):
        if self._is_symbolic:
            rv = random_symbols(expr)[0]
            k = Dummy('k', integer=True)
            expr = expr.subs(rv, k)
            cond = True if not isinstance(expr, (Relational, Logic)) else expr
            func = self.pmf(k) * k if cond != True else self.pmf(k) * expr
            return Sum(Piecewise((func, cond), (S.Zero, True)), (k, self.distribution.low, self.distribution.high)).doit()
        expr = _sympify(expr)
        expr = rv_subs(expr, rvs)
        return FinitePSpace(self.domain, self.distribution).compute_expectation(expr, rvs, **kwargs)

    def probability(self, condition):
        if self._is_symbolic:
            raise NotImplementedError('Currently, probability queries are not supported for random variables with symbolic sized distributions.')
        condition = rv_subs(condition)
        return FinitePSpace(self.domain, self.distribution).probability(condition)

    def conditional_space(self, condition):
        """
        This method is used for transferring the
        computation to probability method because
        conditional space of random variables with
        symbolic dimensions is currently not possible.
        """
        if self._is_symbolic:
            self
        domain = self.where(condition)
        prob = self.probability(condition)
        density = {key: val / prob for key, val in self._density.items() if domain._test(key)}
        return FinitePSpace(domain, density)