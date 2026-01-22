import functools
import math
import numbers
import operator
import weakref
from typing import List
import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (
from torch.nn.functional import pad, softplus
class CumulativeDistributionTransform(Transform):
    """
    Transform via the cumulative distribution function of a probability distribution.

    Args:
        distribution (Distribution): Distribution whose cumulative distribution function to use for
            the transformation.

    Example::

        # Construct a Gaussian copula from a multivariate normal.
        base_dist = MultivariateNormal(
            loc=torch.zeros(2),
            scale_tril=LKJCholesky(2).sample(),
        )
        transform = CumulativeDistributionTransform(Normal(0, 1))
        copula = TransformedDistribution(base_dist, [transform])
    """
    bijective = True
    codomain = constraints.unit_interval
    sign = +1

    def __init__(self, distribution, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.distribution = distribution

    @property
    def domain(self):
        return self.distribution.support

    def _call(self, x):
        return self.distribution.cdf(x)

    def _inverse(self, y):
        return self.distribution.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        return self.distribution.log_prob(x)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CumulativeDistributionTransform(self.distribution, cache_size=cache_size)