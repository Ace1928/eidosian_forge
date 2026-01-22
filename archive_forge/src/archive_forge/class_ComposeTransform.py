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
class ComposeTransform(Transform):
    """
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.
    """

    def __init__(self, parts: List[Transform], cache_size=0):
        if cache_size:
            parts = [part.with_cache(cache_size) for part in parts]
        super().__init__(cache_size=cache_size)
        self.parts = parts

    def __eq__(self, other):
        if not isinstance(other, ComposeTransform):
            return False
        return self.parts == other.parts

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        if not self.parts:
            return constraints.real
        domain = self.parts[0].domain
        event_dim = self.parts[-1].codomain.event_dim
        for part in reversed(self.parts):
            event_dim += part.domain.event_dim - part.codomain.event_dim
            event_dim = max(event_dim, part.domain.event_dim)
        assert event_dim >= domain.event_dim
        if event_dim > domain.event_dim:
            domain = constraints.independent(domain, event_dim - domain.event_dim)
        return domain

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        if not self.parts:
            return constraints.real
        codomain = self.parts[-1].codomain
        event_dim = self.parts[0].domain.event_dim
        for part in self.parts:
            event_dim += part.codomain.event_dim - part.domain.event_dim
            event_dim = max(event_dim, part.codomain.event_dim)
        assert event_dim >= codomain.event_dim
        if event_dim > codomain.event_dim:
            codomain = constraints.independent(codomain, event_dim - codomain.event_dim)
        return codomain

    @lazy_property
    def bijective(self):
        return all((p.bijective for p in self.parts))

    @lazy_property
    def sign(self):
        sign = 1
        for p in self.parts:
            sign = sign * p.sign
        return sign

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = ComposeTransform([p.inv for p in reversed(self.parts)])
            self._inv = weakref.ref(inv)
            inv._inv = weakref.ref(self)
        return inv

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ComposeTransform(self.parts, cache_size=cache_size)

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def log_abs_det_jacobian(self, x, y):
        if not self.parts:
            return torch.zeros_like(x)
        xs = [x]
        for part in self.parts[:-1]:
            xs.append(part(xs[-1]))
        xs.append(y)
        terms = []
        event_dim = self.domain.event_dim
        for part, x, y in zip(self.parts, xs[:-1], xs[1:]):
            terms.append(_sum_rightmost(part.log_abs_det_jacobian(x, y), event_dim - part.domain.event_dim))
            event_dim += part.codomain.event_dim - part.domain.event_dim
        return functools.reduce(operator.add, terms)

    def forward_shape(self, shape):
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape

    def inverse_shape(self, shape):
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        return shape

    def __repr__(self):
        fmt_string = self.__class__.__name__ + '(\n    '
        fmt_string += ',\n    '.join([p.__repr__() for p in self.parts])
        fmt_string += '\n)'
        return fmt_string