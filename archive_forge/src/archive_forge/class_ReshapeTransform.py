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
class ReshapeTransform(Transform):
    """
    Unit Jacobian transform to reshape the rightmost part of a tensor.

    Note that ``in_shape`` and ``out_shape`` must have the same number of
    elements, just as for :meth:`torch.Tensor.reshape`.

    Arguments:
        in_shape (torch.Size): The input event shape.
        out_shape (torch.Size): The output event shape.
    """
    bijective = True

    def __init__(self, in_shape, out_shape, cache_size=0):
        self.in_shape = torch.Size(in_shape)
        self.out_shape = torch.Size(out_shape)
        if self.in_shape.numel() != self.out_shape.numel():
            raise ValueError('in_shape, out_shape have different numbers of elements')
        super().__init__(cache_size=cache_size)

    @constraints.dependent_property
    def domain(self):
        return constraints.independent(constraints.real, len(self.in_shape))

    @constraints.dependent_property
    def codomain(self):
        return constraints.independent(constraints.real, len(self.out_shape))

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ReshapeTransform(self.in_shape, self.out_shape, cache_size=cache_size)

    def _call(self, x):
        batch_shape = x.shape[:x.dim() - len(self.in_shape)]
        return x.reshape(batch_shape + self.out_shape)

    def _inverse(self, y):
        batch_shape = y.shape[:y.dim() - len(self.out_shape)]
        return y.reshape(batch_shape + self.in_shape)

    def log_abs_det_jacobian(self, x, y):
        batch_shape = x.shape[:x.dim() - len(self.in_shape)]
        return x.new_zeros(batch_shape)

    def forward_shape(self, shape):
        if len(shape) < len(self.in_shape):
            raise ValueError('Too few dimensions on input')
        cut = len(shape) - len(self.in_shape)
        if shape[cut:] != self.in_shape:
            raise ValueError(f'Shape mismatch: expected {shape[cut:]} but got {self.in_shape}')
        return shape[:cut] + self.out_shape

    def inverse_shape(self, shape):
        if len(shape) < len(self.out_shape):
            raise ValueError('Too few dimensions on input')
        cut = len(shape) - len(self.out_shape)
        if shape[cut:] != self.out_shape:
            raise ValueError(f'Shape mismatch: expected {shape[cut:]} but got {self.out_shape}')
        return shape[:cut] + self.in_shape