import xarray as xr
from .linalg import (
@xr.register_dataarray_accessor('einops')
class EinopsAccessor:
    """Class that registers accessors for einops functions to the DataArray class."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        try:
            from .einops import rearrange, reduce
            self._rearrange = rearrange
            self._reduce = reduce
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError('`einops` library must be installed in order to use the einops accessor') from err

    def rearrange(self, pattern, pattern_in=None, **kwargs):
        """Call :func:`xarray_einstats.einops.rearrange` on this DataArray."""
        return self._rearrange(self._obj, pattern=pattern, pattern_in=pattern_in, **kwargs)

    def reduce(self, pattern, reduction, pattern_in=None, **kwargs):
        """Call :func:`xarray_einstats.einops.reduce` on this DataArray."""
        return self._reduce(self._obj, pattern=pattern, reduction=reduction, pattern_in=pattern_in, **kwargs)