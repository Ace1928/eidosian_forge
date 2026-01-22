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
class SampleJointNumpy:
    """Returns the sample from numpy of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""
        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        numpy_rv_map = {'MultivariateNormalDistribution': lambda dist, size: rand_state.multivariate_normal(mean=matrix2numpy(dist.mu, float).flatten(), cov=matrix2numpy(dist.sigma, float), size=size), 'MultivariateBetaDistribution': lambda dist, size: rand_state.dirichlet(alpha=list2numpy(dist.alpha, float).flatten(), size=size), 'MultinomialDistribution': lambda dist, size: rand_state.multinomial(n=int(dist.n), pvals=list2numpy(dist.p, float).flatten(), size=size)}
        sample_shape = {'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape, 'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape, 'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape}
        dist_list = numpy_rv_map.keys()
        if dist.__class__.__name__ not in dist_list:
            return None
        samples = numpy_rv_map[dist.__class__.__name__](dist, prod(size))
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))