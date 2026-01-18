import torch
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform

    Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
    where::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = InverseGamma(torch.tensor([2.0]), torch.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    