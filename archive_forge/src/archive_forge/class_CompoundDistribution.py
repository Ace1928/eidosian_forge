from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.symbol import Dummy
from sympy.integrals.integrals import Integral
from sympy.stats.rv import (NamedArgsMixin, random_symbols, _symbol_converter,
from sympy.stats.crv import ContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import DiscreteDistribution, SingleDiscretePSpace
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace
from sympy.stats.crv_types import ContinuousDistributionHandmade
from sympy.stats.drv_types import DiscreteDistributionHandmade
from sympy.stats.frv_types import FiniteDistributionHandmade
class CompoundDistribution(Distribution, NamedArgsMixin):
    """
    Class for Compound Distributions.

    Parameters
    ==========

    dist : Distribution
        Distribution must contain a random parameter

    Examples
    ========

    >>> from sympy.stats.compound_rv import CompoundDistribution
    >>> from sympy.stats.crv_types import NormalDistribution
    >>> from sympy.stats import Normal
    >>> from sympy.abc import x
    >>> X = Normal('X', 2, 4)
    >>> N = NormalDistribution(X, 4)
    >>> C = CompoundDistribution(N)
    >>> C.set
    Interval(-oo, oo)
    >>> C.pdf(x, evaluate=True).simplify()
    exp(-x**2/64 + x/16 - 1/16)/(8*sqrt(pi))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Compound_probability_distribution

    """

    def __new__(cls, dist):
        if not isinstance(dist, (ContinuousDistribution, SingleFiniteDistribution, DiscreteDistribution)):
            message = 'Compound Distribution for %s is not implemented yet' % str(dist)
            raise NotImplementedError(message)
        if not cls._compound_check(dist):
            return dist
        return Basic.__new__(cls, dist)

    @property
    def set(self):
        return self.args[0].set

    @property
    def is_Continuous(self):
        return isinstance(self.args[0], ContinuousDistribution)

    @property
    def is_Finite(self):
        return isinstance(self.args[0], SingleFiniteDistribution)

    @property
    def is_Discrete(self):
        return isinstance(self.args[0], DiscreteDistribution)

    def pdf(self, x, evaluate=False):
        dist = self.args[0]
        randoms = [rv for rv in dist.args if is_random(rv)]
        if isinstance(dist, SingleFiniteDistribution):
            y = Dummy('y', integer=True, negative=False)
            expr = dist.pmf(y)
        else:
            y = Dummy('y')
            expr = dist.pdf(y)
        for rv in randoms:
            expr = self._marginalise(expr, rv, evaluate)
        return Lambda(y, expr)(x)

    def _marginalise(self, expr, rv, evaluate):
        if isinstance(rv.pspace.distribution, SingleFiniteDistribution):
            rv_dens = rv.pspace.distribution.pmf(rv)
        else:
            rv_dens = rv.pspace.distribution.pdf(rv)
        rv_dom = rv.pspace.domain.set
        if rv.pspace.is_Discrete or rv.pspace.is_Finite:
            expr = Sum(expr * rv_dens, (rv, rv_dom._inf, rv_dom._sup))
        else:
            expr = Integral(expr * rv_dens, (rv, rv_dom._inf, rv_dom._sup))
        if evaluate:
            return expr.doit()
        return expr

    @classmethod
    def _compound_check(self, dist):
        """
        Checks if the given distribution contains random parameters.
        """
        randoms = []
        for arg in dist.args:
            randoms.extend(random_symbols(arg))
        if len(randoms) == 0:
            return False
        return True