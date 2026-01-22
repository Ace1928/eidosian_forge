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
class StickBreakingTransform(Transform):
    """
    Transform from unconstrained space to the simplex of one additional
    dimension via a stick-breaking process.

    This transform arises as an iterated sigmoid transform in a stick-breaking
    construction of the `Dirichlet` distribution: the first logit is
    transformed via sigmoid to the first probability and the probability of
    everything else, and then the process recurses.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """
    domain = constraints.real_vector
    codomain = constraints.simplex
    bijective = True

    def __eq__(self, other):
        return isinstance(other, StickBreakingTransform)

    def _call(self, x):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        z = _clipped_sigmoid(x - offset.log())
        z_cumprod = (1 - z).cumprod(-1)
        y = pad(z, [0, 1], value=1) * pad(z_cumprod, [1, 0], value=1)
        return y

    def _inverse(self, y):
        y_crop = y[..., :-1]
        offset = y.shape[-1] - y.new_ones(y_crop.shape[-1]).cumsum(-1)
        sf = 1 - y_crop.cumsum(-1)
        sf = torch.clamp(sf, min=torch.finfo(y.dtype).tiny)
        x = y_crop.log() - sf.log() + offset.log()
        return x

    def log_abs_det_jacobian(self, x, y):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        x = x - offset.log()
        detJ = (-x + F.logsigmoid(x) + y[..., :-1].log()).sum(-1)
        return detJ

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError('Too few dimensions on input')
        return shape[:-1] + (shape[-1] + 1,)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError('Too few dimensions on input')
        return shape[:-1] + (shape[-1] - 1,)