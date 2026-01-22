from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core._aggregations import (
from xarray.core.groupby import DataArrayGroupByBase, DatasetGroupByBase, GroupBy
from xarray.core.types import Dims, InterpOptions, T_Xarray
class DataArrayResample(Resample['DataArray'], DataArrayGroupByBase, DataArrayResampleAggregations):
    """DataArrayGroupBy object specialized to time resampling operations over a
    specified dimension
    """

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> DataArray:
        """Reduce the items in this group by applying `func` along the
        pre-defined resampling dimension.

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        return super().reduce(func=func, dim=dim, axis=axis, keep_attrs=keep_attrs, keepdims=keepdims, shortcut=shortcut, **kwargs)

    def map(self, func: Callable[..., Any], args: tuple[Any, ...]=(), shortcut: bool | None=False, **kwargs: Any) -> DataArray:
        """Apply a function to each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:

            (1) The action of `func` does not depend on any of the array
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.

            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray
            The result of splitting, applying and combining this array.
        """
        return self._map_maybe_warn(func, args, shortcut, warn_squeeze=True, **kwargs)

    def _map_maybe_warn(self, func: Callable[..., Any], args: tuple[Any, ...]=(), shortcut: bool | None=False, warn_squeeze: bool=True, **kwargs: Any) -> DataArray:
        combined = super()._map_maybe_warn(func, shortcut=shortcut, args=args, warn_squeeze=warn_squeeze, **kwargs)
        if self._dim in combined.coords:
            combined = combined.drop_vars([self._dim])
        if RESAMPLE_DIM in combined.dims:
            combined = combined.rename({RESAMPLE_DIM: self._dim})
        return combined

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataArrayResample.map
        """
        warnings.warn('Resample.apply may be deprecated in the future. Using Resample.map is encouraged', PendingDeprecationWarning, stacklevel=2)
        return self.map(func=func, shortcut=shortcut, args=args, **kwargs)

    def asfreq(self) -> DataArray:
        """Return values of original object at the new up-sampling frequency;
        essentially a re-index with new times set to NaN.

        Returns
        -------
        resampled : DataArray
        """
        self._obj = self._drop_coords()
        return self.mean(None if self._dim is None else [self._dim])