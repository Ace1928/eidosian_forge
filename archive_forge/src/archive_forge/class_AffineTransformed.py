from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
class AffineTransformed(TransformedDistribution):

    def __init__(self, base_distribution: Distribution, loc=None, scale=None, event_dim=0):
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc
        super().__init__(base_distribution, [AffineTransform(loc=self.loc, scale=self.scale, event_dim=event_dim)])

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale ** 2

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()