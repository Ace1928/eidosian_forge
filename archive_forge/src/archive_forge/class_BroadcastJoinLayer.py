from __future__ import annotations
import functools
import math
import operator
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
import tlz as toolz
from tlz.curried import map
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise_token
from dask.core import flatten
from dask.highlevelgraph import Layer
from dask.utils import apply, cached_cumsum, concrete, insert
class BroadcastJoinLayer(Layer):
    """Broadcast-based Join Layer

    High-level graph layer for a join operation requiring the
    smaller collection to be broadcasted to every partition of
    the larger collection.

    Parameters
    ----------
    name : str
        Name of new (joined) output collection.
    lhs_name: string
        "Left" DataFrame collection to join.
    lhs_npartitions: int
        Number of partitions in "left" DataFrame collection.
    rhs_name: string
        "Right" DataFrame collection to join.
    rhs_npartitions: int
        Number of partitions in "right" DataFrame collection.
    parts_out : list of int (optional)
        List of required output-partition indices.
    annotations : dict (optional)
        Layer annotations.
    **merge_kwargs : **dict
        Keyword arguments to be passed to chunkwise merge func.
    """

    def __init__(self, name, npartitions, lhs_name, lhs_npartitions, rhs_name, rhs_npartitions, parts_out=None, annotations=None, left_on=None, right_on=None, **merge_kwargs):
        super().__init__(annotations=annotations)
        self.name = name
        self.npartitions = npartitions
        self.lhs_name = lhs_name
        self.lhs_npartitions = lhs_npartitions
        self.rhs_name = rhs_name
        self.rhs_npartitions = rhs_npartitions
        self.parts_out = parts_out or set(range(self.npartitions))
        self.left_on = tuple(left_on) if isinstance(left_on, list) else left_on
        self.right_on = tuple(right_on) if isinstance(right_on, list) else right_on
        self.merge_kwargs = merge_kwargs
        self.how = self.merge_kwargs.get('how')
        self.merge_kwargs['left_on'] = self.left_on
        self.merge_kwargs['right_on'] = self.right_on

    def get_output_keys(self):
        return {(self.name, part) for part in self.parts_out}

    def __repr__(self):
        return "BroadcastJoinLayer<name='{}', how={}, lhs={}, rhs={}>".format(self.name, self.how, self.lhs_name, self.rhs_name)

    def is_materialized(self):
        return hasattr(self, '_cached_dict')

    @property
    def _dict(self):
        """Materialize full dict representation"""
        if hasattr(self, '_cached_dict'):
            return self._cached_dict
        else:
            dsk = self._construct_graph()
            self._cached_dict = dsk
        return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def _keys_to_parts(self, keys):
        """Simple utility to convert keys to partition indices."""
        parts = set()
        for key in keys:
            try:
                _name, _part = key
            except ValueError:
                continue
            if _name != self.name:
                continue
            parts.add(_part)
        return parts

    @property
    def _broadcast_plan(self):
        if self.lhs_npartitions < self.rhs_npartitions:
            return (self.lhs_name, self.lhs_npartitions, self.rhs_name, self.right_on)
        else:
            return (self.rhs_name, self.rhs_npartitions, self.lhs_name, self.left_on)

    def _cull_dependencies(self, keys, parts_out=None):
        """Determine the necessary dependencies to produce `keys`.

        For a broadcast join, output partitions always depend on
        all partitions of the broadcasted collection, but only one
        partition of the "other" collection.
        """
        bcast_name, bcast_size, other_name = self._broadcast_plan[:3]
        deps = defaultdict(set)
        parts_out = parts_out or self._keys_to_parts(keys)
        for part in parts_out:
            deps[self.name, part] |= {(bcast_name, i) for i in range(bcast_size)}
            deps[self.name, part] |= {(other_name, part)}
        return deps

    def _cull(self, parts_out):
        return BroadcastJoinLayer(self.name, self.npartitions, self.lhs_name, self.lhs_npartitions, self.rhs_name, self.rhs_npartitions, annotations=self.annotations, parts_out=parts_out, **self.merge_kwargs)

    def cull(self, keys, all_keys):
        """Cull a BroadcastJoinLayer HighLevelGraph layer.

        The underlying graph will only include the necessary
        tasks to produce the keys (indices) included in `parts_out`.
        Therefore, "culling" the layer only requires us to reset this
        parameter.
        """
        parts_out = self._keys_to_parts(keys)
        culled_deps = self._cull_dependencies(keys, parts_out=parts_out)
        if parts_out != set(self.parts_out):
            culled_layer = self._cull(parts_out)
            return (culled_layer, culled_deps)
        else:
            return (self, culled_deps)

    def _construct_graph(self, deserializing=False):
        """Construct graph for a broadcast join operation."""
        inter_name = 'inter-' + self.name
        split_name = 'split-' + self.name
        if deserializing:
            split_partition_func = CallableLazyImport('dask.dataframe.multi._split_partition')
            concat_func = CallableLazyImport('dask.dataframe.multi._concat_wrapper')
            merge_chunk_func = CallableLazyImport('dask.dataframe.multi._merge_chunk_wrapper')
        else:
            from dask.dataframe.multi import _concat_wrapper as concat_func
            from dask.dataframe.multi import _merge_chunk_wrapper as merge_chunk_func
            from dask.dataframe.multi import _split_partition as split_partition_func
        bcast_name, bcast_size, other_name, other_on = self._broadcast_plan
        bcast_side = 'left' if self.lhs_npartitions < self.rhs_npartitions else 'right'
        dsk = {}
        for i in self.parts_out:
            if self.how != 'inner':
                dsk[split_name, i] = (split_partition_func, (other_name, i), other_on, bcast_size)
            _concat_list = []
            for j in range(bcast_size):
                _merge_args = [(operator.getitem, (split_name, i), j) if self.how != 'inner' else (other_name, i), (bcast_name, j)]
                if bcast_side == 'left':
                    _merge_args.reverse()
                inter_key = (inter_name, i, j)
                dsk[inter_key] = (apply, merge_chunk_func, _merge_args, self.merge_kwargs)
                _concat_list.append(inter_key)
            dsk[self.name, i] = (concat_func, _concat_list)
        return dsk