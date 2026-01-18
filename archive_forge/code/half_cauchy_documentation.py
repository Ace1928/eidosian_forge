import math
import torch
from torch import inf
from torch.distributions import constraints
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform

    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    