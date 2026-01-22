from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
class Aligner(Generic[T_Alignable]):
    """Implements all the complex logic for the re-indexing and alignment of Xarray
    objects.

    For internal use only, not public API.
    Usage:

    aligner = Aligner(*objects, **kwargs)
    aligner.align()
    aligned_objects = aligner.results

    """
    objects: tuple[T_Alignable, ...]
    results: tuple[T_Alignable, ...]
    objects_matching_indexes: tuple[dict[MatchingIndexKey, Index], ...]
    join: str
    exclude_dims: frozenset[Hashable]
    exclude_vars: frozenset[Hashable]
    copy: bool
    fill_value: Any
    sparse: bool
    indexes: dict[MatchingIndexKey, Index]
    index_vars: dict[MatchingIndexKey, dict[Hashable, Variable]]
    all_indexes: dict[MatchingIndexKey, list[Index]]
    all_index_vars: dict[MatchingIndexKey, list[dict[Hashable, Variable]]]
    aligned_indexes: dict[MatchingIndexKey, Index]
    aligned_index_vars: dict[MatchingIndexKey, dict[Hashable, Variable]]
    reindex: dict[MatchingIndexKey, bool]
    reindex_kwargs: dict[str, Any]
    unindexed_dim_sizes: dict[Hashable, set]
    new_indexes: Indexes[Index]

    def __init__(self, objects: Iterable[T_Alignable], join: str='inner', indexes: Mapping[Any, Any] | None=None, exclude_dims: str | Iterable[Hashable]=frozenset(), exclude_vars: Iterable[Hashable]=frozenset(), method: str | None=None, tolerance: int | float | Iterable[int | float] | None=None, copy: bool=True, fill_value: Any=dtypes.NA, sparse: bool=False):
        self.objects = tuple(objects)
        self.objects_matching_indexes = ()
        if join not in ['inner', 'outer', 'override', 'exact', 'left', 'right']:
            raise ValueError(f'invalid value for join: {join}')
        self.join = join
        self.copy = copy
        self.fill_value = fill_value
        self.sparse = sparse
        if method is None and tolerance is None:
            self.reindex_kwargs = {}
        else:
            self.reindex_kwargs = {'method': method, 'tolerance': tolerance}
        if isinstance(exclude_dims, str):
            exclude_dims = [exclude_dims]
        self.exclude_dims = frozenset(exclude_dims)
        self.exclude_vars = frozenset(exclude_vars)
        if indexes is None:
            indexes = {}
        self.indexes, self.index_vars = self._normalize_indexes(indexes)
        self.all_indexes = {}
        self.all_index_vars = {}
        self.unindexed_dim_sizes = {}
        self.aligned_indexes = {}
        self.aligned_index_vars = {}
        self.reindex = {}
        self.results = tuple()

    def _normalize_indexes(self, indexes: Mapping[Any, Any | T_DuckArray]) -> tuple[NormalizedIndexes, NormalizedIndexVars]:
        """Normalize the indexes/indexers used for re-indexing or alignment.

        Return dictionaries of xarray Index objects and coordinate variables
        such that we can group matching indexes based on the dictionary keys.

        """
        if isinstance(indexes, Indexes):
            xr_variables = dict(indexes.variables)
        else:
            xr_variables = {}
        xr_indexes: dict[Hashable, Index] = {}
        for k, idx in indexes.items():
            if not isinstance(idx, Index):
                if getattr(idx, 'dims', (k,)) != (k,):
                    raise ValueError(f"Indexer has dimensions {idx.dims} that are different from that to be indexed along '{k}'")
                data: T_DuckArray = as_compatible_data(idx)
                pd_idx = safe_cast_to_index(data)
                pd_idx.name = k
                if isinstance(pd_idx, pd.MultiIndex):
                    idx = PandasMultiIndex(pd_idx, k)
                else:
                    idx = PandasIndex(pd_idx, k, coord_dtype=data.dtype)
                xr_variables.update(idx.create_variables())
            xr_indexes[k] = idx
        normalized_indexes = {}
        normalized_index_vars = {}
        for idx, index_vars in Indexes(xr_indexes, xr_variables).group_by_index():
            coord_names_and_dims = []
            all_dims: set[Hashable] = set()
            for name, var in index_vars.items():
                dims = var.dims
                coord_names_and_dims.append((name, dims))
                all_dims.update(dims)
            exclude_dims = all_dims & self.exclude_dims
            if exclude_dims == all_dims:
                continue
            elif exclude_dims:
                excl_dims_str = ', '.join((str(d) for d in exclude_dims))
                incl_dims_str = ', '.join((str(d) for d in all_dims - exclude_dims))
                raise ValueError(f'cannot exclude dimension(s) {excl_dims_str} from alignment because these are used by an index together with non-excluded dimensions {incl_dims_str}')
            key = (tuple(coord_names_and_dims), type(idx))
            normalized_indexes[key] = idx
            normalized_index_vars[key] = index_vars
        return (normalized_indexes, normalized_index_vars)

    def find_matching_indexes(self) -> None:
        all_indexes: dict[MatchingIndexKey, list[Index]]
        all_index_vars: dict[MatchingIndexKey, list[dict[Hashable, Variable]]]
        all_indexes_dim_sizes: dict[MatchingIndexKey, dict[Hashable, set]]
        objects_matching_indexes: list[dict[MatchingIndexKey, Index]]
        all_indexes = defaultdict(list)
        all_index_vars = defaultdict(list)
        all_indexes_dim_sizes = defaultdict(lambda: defaultdict(set))
        objects_matching_indexes = []
        for obj in self.objects:
            obj_indexes, obj_index_vars = self._normalize_indexes(obj.xindexes)
            objects_matching_indexes.append(obj_indexes)
            for key, idx in obj_indexes.items():
                all_indexes[key].append(idx)
            for key, index_vars in obj_index_vars.items():
                all_index_vars[key].append(index_vars)
                for dim, size in calculate_dimensions(index_vars).items():
                    all_indexes_dim_sizes[key][dim].add(size)
        self.objects_matching_indexes = tuple(objects_matching_indexes)
        self.all_indexes = all_indexes
        self.all_index_vars = all_index_vars
        if self.join == 'override':
            for dim_sizes in all_indexes_dim_sizes.values():
                for dim, sizes in dim_sizes.items():
                    if len(sizes) > 1:
                        raise ValueError(f"cannot align objects with join='override' with matching indexes along dimension {dim!r} that don't have the same size")

    def find_matching_unindexed_dims(self) -> None:
        unindexed_dim_sizes = defaultdict(set)
        for obj in self.objects:
            for dim in obj.dims:
                if dim not in self.exclude_dims and dim not in obj.xindexes.dims:
                    unindexed_dim_sizes[dim].add(obj.sizes[dim])
        self.unindexed_dim_sizes = unindexed_dim_sizes

    def assert_no_index_conflict(self) -> None:
        """Check for uniqueness of both coordinate and dimension names across all sets
        of matching indexes.

        We need to make sure that all indexes used for re-indexing or alignment
        are fully compatible and do not conflict each other.

        Note: perhaps we could choose less restrictive constraints and instead
        check for conflicts among the dimension (position) indexers returned by
        `Index.reindex_like()` for each matching pair of object index / aligned
        index?
        (ref: https://github.com/pydata/xarray/issues/1603#issuecomment-442965602)

        """
        matching_keys = set(self.all_indexes) | set(self.indexes)
        coord_count: dict[Hashable, int] = defaultdict(int)
        dim_count: dict[Hashable, int] = defaultdict(int)
        for coord_names_dims, _ in matching_keys:
            dims_set: set[Hashable] = set()
            for name, dims in coord_names_dims:
                coord_count[name] += 1
                dims_set.update(dims)
            for dim in dims_set:
                dim_count[dim] += 1
        for count, msg in [(coord_count, 'coordinates'), (dim_count, 'dimensions')]:
            dup = {k: v for k, v in count.items() if v > 1}
            if dup:
                items_msg = ', '.join((f'{k!r} ({v} conflicting indexes)' for k, v in dup.items()))
                raise ValueError(f"cannot re-index or align objects with conflicting indexes found for the following {msg}: {items_msg}\nConflicting indexes may occur when\n- they relate to different sets of coordinate and/or dimension names\n- they don't have the same type\n- they may be used to reindex data along common dimensions")

    def _need_reindex(self, dim, cmp_indexes) -> bool:
        """Whether or not we need to reindex variables for a set of
        matching indexes.

        We don't reindex when all matching indexes are equal for two reasons:
        - It's faster for the usual case (already aligned objects).
        - It ensures it's possible to do operations that don't require alignment
          on indexes with duplicate values (which cannot be reindexed with
          pandas). This is useful, e.g., for overwriting such duplicate indexes.

        """
        if not indexes_all_equal(cmp_indexes):
            return True
        unindexed_dims_sizes = {}
        for d in dim:
            if d in self.unindexed_dim_sizes:
                sizes = self.unindexed_dim_sizes[d]
                if len(sizes) > 1:
                    return True
                else:
                    unindexed_dims_sizes[d] = next(iter(sizes))
        if unindexed_dims_sizes:
            indexed_dims_sizes = {}
            for cmp in cmp_indexes:
                index_vars = cmp[1]
                for var in index_vars.values():
                    indexed_dims_sizes.update(var.sizes)
            for d, size in unindexed_dims_sizes.items():
                if indexed_dims_sizes.get(d, -1) != size:
                    return True
        return False

    def _get_index_joiner(self, index_cls) -> Callable:
        if self.join in ['outer', 'inner']:
            return functools.partial(functools.reduce, functools.partial(index_cls.join, how=self.join))
        elif self.join == 'left':
            return operator.itemgetter(0)
        elif self.join == 'right':
            return operator.itemgetter(-1)
        elif self.join == 'override':
            return operator.itemgetter(0)
        else:
            return lambda _: None

    def align_indexes(self) -> None:
        """Compute all aligned indexes and their corresponding coordinate variables."""
        aligned_indexes = {}
        aligned_index_vars = {}
        reindex = {}
        new_indexes = {}
        new_index_vars = {}
        for key, matching_indexes in self.all_indexes.items():
            matching_index_vars = self.all_index_vars[key]
            dims = {d for coord in matching_index_vars[0].values() for d in coord.dims}
            index_cls = key[1]
            if self.join == 'override':
                joined_index = matching_indexes[0]
                joined_index_vars = matching_index_vars[0]
                need_reindex = False
            elif key in self.indexes:
                joined_index = self.indexes[key]
                joined_index_vars = self.index_vars[key]
                cmp_indexes = list(zip([joined_index] + matching_indexes, [joined_index_vars] + matching_index_vars))
                need_reindex = self._need_reindex(dims, cmp_indexes)
            else:
                if len(matching_indexes) > 1:
                    need_reindex = self._need_reindex(dims, list(zip(matching_indexes, matching_index_vars)))
                else:
                    need_reindex = False
                if need_reindex:
                    if self.join == 'exact':
                        raise ValueError("cannot align objects with join='exact' where index/labels/sizes are not equal along these coordinates (dimensions): " + ', '.join((f'{name!r} {dims!r}' for name, dims in key[0])))
                    joiner = self._get_index_joiner(index_cls)
                    joined_index = joiner(matching_indexes)
                    if self.join == 'left':
                        joined_index_vars = matching_index_vars[0]
                    elif self.join == 'right':
                        joined_index_vars = matching_index_vars[-1]
                    else:
                        joined_index_vars = joined_index.create_variables()
                else:
                    joined_index = matching_indexes[0]
                    joined_index_vars = matching_index_vars[0]
            reindex[key] = need_reindex
            aligned_indexes[key] = joined_index
            aligned_index_vars[key] = joined_index_vars
            for name, var in joined_index_vars.items():
                new_indexes[name] = joined_index
                new_index_vars[name] = var
        for key, idx in self.indexes.items():
            if key not in aligned_indexes:
                index_vars = self.index_vars[key]
                reindex[key] = False
                aligned_indexes[key] = idx
                aligned_index_vars[key] = index_vars
                for name, var in index_vars.items():
                    new_indexes[name] = idx
                    new_index_vars[name] = var
        self.aligned_indexes = aligned_indexes
        self.aligned_index_vars = aligned_index_vars
        self.reindex = reindex
        self.new_indexes = Indexes(new_indexes, new_index_vars)

    def assert_unindexed_dim_sizes_equal(self) -> None:
        for dim, sizes in self.unindexed_dim_sizes.items():
            index_size = self.new_indexes.dims.get(dim)
            if index_size is not None:
                sizes.add(index_size)
                add_err_msg = f' (note: an index is found along that dimension with size={index_size!r})'
            else:
                add_err_msg = ''
            if len(sizes) > 1:
                raise ValueError(f'cannot reindex or align along dimension {dim!r} because of conflicting dimension sizes: {sizes!r}' + add_err_msg)

    def override_indexes(self) -> None:
        objects = list(self.objects)
        for i, obj in enumerate(objects[1:]):
            new_indexes = {}
            new_variables = {}
            matching_indexes = self.objects_matching_indexes[i + 1]
            for key, aligned_idx in self.aligned_indexes.items():
                obj_idx = matching_indexes.get(key)
                if obj_idx is not None:
                    for name, var in self.aligned_index_vars[key].items():
                        new_indexes[name] = aligned_idx
                        new_variables[name] = var.copy(deep=self.copy)
            objects[i + 1] = obj._overwrite_indexes(new_indexes, new_variables)
        self.results = tuple(objects)

    def _get_dim_pos_indexers(self, matching_indexes: dict[MatchingIndexKey, Index]) -> dict[Hashable, Any]:
        dim_pos_indexers = {}
        for key, aligned_idx in self.aligned_indexes.items():
            obj_idx = matching_indexes.get(key)
            if obj_idx is not None:
                if self.reindex[key]:
                    indexers = obj_idx.reindex_like(aligned_idx, **self.reindex_kwargs)
                    dim_pos_indexers.update(indexers)
        return dim_pos_indexers

    def _get_indexes_and_vars(self, obj: T_Alignable, matching_indexes: dict[MatchingIndexKey, Index]) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
        new_indexes = {}
        new_variables = {}
        for key, aligned_idx in self.aligned_indexes.items():
            index_vars = self.aligned_index_vars[key]
            obj_idx = matching_indexes.get(key)
            if obj_idx is None:
                index_vars_dims = {d for var in index_vars.values() for d in var.dims}
                if index_vars_dims <= set(obj.dims):
                    obj_idx = aligned_idx
            if obj_idx is not None:
                for name, var in index_vars.items():
                    new_indexes[name] = aligned_idx
                    new_variables[name] = var.copy(deep=self.copy)
        return (new_indexes, new_variables)

    def _reindex_one(self, obj: T_Alignable, matching_indexes: dict[MatchingIndexKey, Index]) -> T_Alignable:
        new_indexes, new_variables = self._get_indexes_and_vars(obj, matching_indexes)
        dim_pos_indexers = self._get_dim_pos_indexers(matching_indexes)
        return obj._reindex_callback(self, dim_pos_indexers, new_variables, new_indexes, self.fill_value, self.exclude_dims, self.exclude_vars)

    def reindex_all(self) -> None:
        self.results = tuple((self._reindex_one(obj, matching_indexes) for obj, matching_indexes in zip(self.objects, self.objects_matching_indexes)))

    def align(self) -> None:
        if not self.indexes and len(self.objects) == 1:
            obj, = self.objects
            self.results = (obj.copy(deep=self.copy),)
            return
        self.find_matching_indexes()
        self.find_matching_unindexed_dims()
        self.assert_no_index_conflict()
        self.align_indexes()
        self.assert_unindexed_dim_sizes_equal()
        if self.join == 'override':
            self.override_indexes()
        elif self.join == 'exact' and (not self.copy):
            self.results = self.objects
        else:
            self.reindex_all()