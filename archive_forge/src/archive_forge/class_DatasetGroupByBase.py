from __future__ import annotations
import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args
class DatasetGroupByBase(GroupBy['Dataset'], DatasetGroupbyArithmetic):
    __slots__ = ()
    _dims: Frozen[Hashable, int] | None

    @property
    def dims(self) -> Frozen[Hashable, int]:
        if self._dims is None:
            grouper, = self.groupers
            index = _maybe_squeeze_indices(self._group_indices[0], self._squeeze, grouper, warn=True)
            self._dims = self._obj.isel({self._group_dim: index}).dims
        return FrozenMappingWarningOnValuesAccess(self._dims)

    def map(self, func: Callable[..., Dataset], args: tuple[Any, ...]=(), shortcut: bool | None=None, **kwargs: Any) -> Dataset:
        """Apply a function to each Dataset in the group and concatenate them
        together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each sub-dataset.
        args : tuple, optional
            Positional arguments to pass to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset
            The result of splitting, applying and combining this dataset.
        """
        return self._map_maybe_warn(func, args, shortcut, warn_squeeze=True, **kwargs)

    def _map_maybe_warn(self, func: Callable[..., Dataset], args: tuple[Any, ...]=(), shortcut: bool | None=None, warn_squeeze: bool=False, **kwargs: Any) -> Dataset:
        applied = (func(ds, *args, **kwargs) for ds in self._iter_grouped(warn_squeeze))
        return self._combine(applied)

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DatasetGroupBy.map
        """
        warnings.warn('GroupBy.apply may be deprecated in the future. Using GroupBy.map is encouraged', PendingDeprecationWarning, stacklevel=2)
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        combined = concat(applied, dim)
        grouper, = self.groupers
        combined = _maybe_reorder(combined, dim, positions, N=grouper.group.size)
        if coord is not None and dim not in applied_example.dims:
            index, index_vars = create_default_index_implicit(coord)
            indexes = {k: index for k in index_vars}
            combined = combined._overwrite_indexes(indexes, index_vars)
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : ..., str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default apply over the
            groupby dimension, with "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = [self._group_dim]
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        def reduce_dataset(ds: Dataset) -> Dataset:
            return ds.reduce(func=func, dim=dim, axis=axis, keep_attrs=keep_attrs, keepdims=keepdims, **kwargs)
        check_reduce_dims(dim, self.dims)
        return self.map(reduce_dataset)

    def _reduce_without_squeeze_warn(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : ..., str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default apply over the
            groupby dimension, with "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = [self._group_dim]
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        def reduce_dataset(ds: Dataset) -> Dataset:
            return ds.reduce(func=func, dim=dim, axis=axis, keep_attrs=keep_attrs, keepdims=keepdims, **kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The `squeeze` kwarg')
            check_reduce_dims(dim, self.dims)
        return self._map_maybe_warn(reduce_dataset, warn_squeeze=False)

    def assign(self, **kwargs: Any) -> Dataset:
        """Assign data variables by group.

        See Also
        --------
        Dataset.assign
        """
        return self.map(lambda ds: ds.assign(**kwargs))