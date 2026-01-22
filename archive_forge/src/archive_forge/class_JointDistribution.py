from math import prod
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.sets.sets import ProductSet
from sympy.tensor.indexed import Indexed
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation
from sympy.core.containers import Tuple
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices import ImmutableMatrix, matrix2numpy, list2numpy
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import (ProductPSpace, NamedArgsMixin, Distribution,
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent
from sympy.external import import_module
class JointDistribution(Distribution, NamedArgsMixin):
    """
    Represented by the random variables part of the joint distribution.
    Contains methods for PDF, CDF, sampling, marginal densities, etc.
    """
    _argnames = ('pdf',)

    def __new__(cls, *args):
        args = list(map(sympify, args))
        for i in range(len(args)):
            if isinstance(args[i], list):
                args[i] = ImmutableMatrix(args[i])
        return Basic.__new__(cls, *args)

    @property
    def domain(self):
        return ProductDomain(self.symbols)

    @property
    def pdf(self):
        return self.density.args[1]

    def cdf(self, other):
        if not isinstance(other, dict):
            raise ValueError('%s should be of type dict, got %s' % (other, type(other)))
        rvs = other.keys()
        _set = self.domain.set.sets
        expr = self.pdf(tuple((i.args[0] for i in self.symbols)))
        for i in range(len(other)):
            if rvs[i].is_Continuous:
                density = Integral(expr, (rvs[i], _set[i].inf, other[rvs[i]]))
            elif rvs[i].is_Discrete:
                density = Sum(expr, (rvs[i], _set[i].inf, other[rvs[i]]))
        return density

    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """
        libraries = ('scipy', 'numpy', 'pymc3', 'pymc')
        if library not in libraries:
            raise NotImplementedError('Sampling from %s is not supported yet.' % str(library))
        if not import_module(library):
            raise ValueError('Failed to import %s' % library)
        samps = _get_sample_class_jrv[library](self, size, seed=seed)
        if samps is not None:
            return samps
        raise NotImplementedError('Sampling for %s is not currently implemented from %s' % (self.__class__.__name__, library))

    def __call__(self, *args):
        return self.pdf(*args)