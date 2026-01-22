from __future__ import annotations
import warnings
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
class _CachedAccessor:
    """Custom property-like object (descriptor) for caching accessors."""

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor
        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}
        try:
            return cache[self._name]
        except KeyError:
            pass
        try:
            accessor_obj = self._accessor(obj)
        except AttributeError:
            raise RuntimeError(f'error initializing {self._name!r} accessor.')
        cache[self._name] = accessor_obj
        return accessor_obj