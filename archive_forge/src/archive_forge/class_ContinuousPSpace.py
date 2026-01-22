from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda, PoleError
from sympy.core.numbers import (I, nan, oo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Union)
from sympy.solvers.solveset import solveset
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats.rv import (RandomDomain, SingleDomain, ConditionalDomain, is_random,
class ContinuousPSpace(PSpace):
    """ Continuous Probability Space

    Represents the likelihood of an event space defined over a continuum.

    Represented with a ContinuousDomain and a PDF (Lambda-Like)
    """
    is_Continuous = True
    is_real = True

    @property
    def pdf(self):
        return self.density(*self.domain.symbols)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        if rvs is None:
            rvs = self.values
        else:
            rvs = frozenset(rvs)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})
        domain_symbols = frozenset((rv.symbol for rv in rvs))
        return self.domain.compute_expectation(self.pdf * expr, domain_symbols, **kwargs)

    def compute_density(self, expr, **kwargs):
        if expr in self.values:
            randomsymbols = tuple(set(self.values) - frozenset([expr]))
            symbols = tuple((rs.symbol for rs in randomsymbols))
            pdf = self.domain.compute_expectation(self.pdf, symbols, **kwargs)
            return Lambda(expr.symbol, pdf)
        z = Dummy('z', real=True)
        return Lambda(z, self.compute_expectation(DiracDelta(expr - z), **kwargs))

    @cacheit
    def compute_cdf(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise ValueError('CDF not well defined on multivariate expressions')
        d = self.compute_density(expr, **kwargs)
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.domain.set.start
        cdf = integrate(d(x), (x, left_bound, z), **kwargs)
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    @cacheit
    def compute_characteristic_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError('Characteristic function of multivariate expressions not implemented')
        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        cf = integrate(exp(I * t * x) * d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, cf)

    @cacheit
    def compute_moment_generating_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError('Moment generating function of multivariate expressions not implemented')
        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        mgf = integrate(exp(t * x) * d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, mgf)

    @cacheit
    def compute_quantile(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise ValueError('Quantile not well defined on multivariate expressions')
        d = self.compute_cdf(expr, **kwargs)
        x = Dummy('x', real=True)
        p = Dummy('p', positive=True)
        quantile = solveset(d(x) - p, x, self.set)
        return Lambda(p, quantile)

    def probability(self, condition, **kwargs):
        z = Dummy('z', real=True)
        cond_inv = False
        if isinstance(condition, Ne):
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        try:
            domain = self.where(condition)
            rv = [rv for rv in self.values if rv.symbol == domain.symbol][0]
            pdf = self.compute_density(rv, **kwargs)
            if domain.set is S.EmptySet or isinstance(domain.set, FiniteSet):
                return S.Zero if not cond_inv else S.One
            if isinstance(domain.set, Union):
                return sum((Integral(pdf(z), (z, subset), **kwargs) for subset in domain.set.args if isinstance(subset, Interval)))
            return Integral(pdf(z), (z, domain.set), **kwargs)
        except NotImplementedError:
            from sympy.stats.rv import density
            expr = condition.lhs - condition.rhs
            if not is_random(expr):
                dens = self.density
                comp = condition.rhs
            else:
                dens = density(expr, **kwargs)
                comp = 0
            if not isinstance(dens, ContinuousDistribution):
                from sympy.stats.crv_types import ContinuousDistributionHandmade
                dens = ContinuousDistributionHandmade(dens, set=self.domain.set)
            space = SingleContinuousPSpace(z, dens)
            result = space.probability(condition.__class__(space.value, comp))
            return result if not cond_inv else S.One - result

    def where(self, condition):
        rvs = frozenset(random_symbols(condition))
        if not (len(rvs) == 1 and rvs.issubset(self.values)):
            raise NotImplementedError('Multiple continuous random variables not supported')
        rv = tuple(rvs)[0]
        interval = reduce_rational_inequalities_wrap(condition, rv)
        interval = interval.intersect(self.domain.set)
        return SingleContinuousDomain(rv.symbol, interval)

    def conditional_space(self, condition, normalize=True, **kwargs):
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        domain = ConditionalContinuousDomain(self.domain, condition)
        if normalize:
            replacement = {rv: Dummy(str(rv)) for rv in self.symbols}
            norm = domain.compute_expectation(self.pdf, **kwargs)
            pdf = self.pdf / norm.xreplace(replacement)
            density = Lambda(tuple(domain.symbols), pdf)
        return ContinuousPSpace(domain, density)