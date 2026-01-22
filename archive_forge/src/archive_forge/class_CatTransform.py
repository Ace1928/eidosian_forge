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
class CatTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`, of length `lengths[dim]`,
    in a way compatible with :func:`torch.cat`.

    Example::

       x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
       x = torch.cat([x0, x0], dim=0)
       t0 = CatTransform([ExpTransform(), identity_transform], dim=0, lengths=[10, 10])
       t = CatTransform([t0, t0], dim=0, lengths=[20, 20])
       y = t(x)
    """
    transforms: List[Transform]

    def __init__(self, tseq, dim=0, lengths=None, cache_size=0):
        assert all((isinstance(t, Transform) for t in tseq))
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        if lengths is None:
            lengths = [1] * len(self.transforms)
        self.lengths = list(lengths)
        assert len(self.lengths) == len(self.transforms)
        self.dim = dim

    @lazy_property
    def event_dim(self):
        return max((t.event_dim for t in self.transforms))

    @lazy_property
    def length(self):
        return sum(self.lengths)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CatTransform(self.transforms, self.dim, self.lengths, cache_size)

    def _call(self, x):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == self.length
        yslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslices.append(trans(xslice))
            start = start + length
        return torch.cat(yslices, dim=self.dim)

    def _inverse(self, y):
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == self.length
        xslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            yslice = y.narrow(self.dim, start, length)
            xslices.append(trans.inv(yslice))
            start = start + length
        return torch.cat(xslices, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        assert -x.dim() <= self.dim < x.dim()
        assert x.size(self.dim) == self.length
        assert -y.dim() <= self.dim < y.dim()
        assert y.size(self.dim) == self.length
        logdetjacs = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslice = y.narrow(self.dim, start, length)
            logdetjac = trans.log_abs_det_jacobian(xslice, yslice)
            if trans.event_dim < self.event_dim:
                logdetjac = _sum_rightmost(logdetjac, self.event_dim - trans.event_dim)
            logdetjacs.append(logdetjac)
            start = start + length
        dim = self.dim
        if dim >= 0:
            dim = dim - x.dim()
        dim = dim + self.event_dim
        if dim < 0:
            return torch.cat(logdetjacs, dim=dim)
        else:
            return sum(logdetjacs)

    @property
    def bijective(self):
        return all((t.bijective for t in self.transforms))

    @constraints.dependent_property
    def domain(self):
        return constraints.cat([t.domain for t in self.transforms], self.dim, self.lengths)

    @constraints.dependent_property
    def codomain(self):
        return constraints.cat([t.codomain for t in self.transforms], self.dim, self.lengths)