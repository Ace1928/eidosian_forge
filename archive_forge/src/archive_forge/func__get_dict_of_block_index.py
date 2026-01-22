import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def _get_dict_of_block_index(self, axis, indices, are_indices_sorted=False):
    """
        Convert indices to an ordered dict mapping partition (or block) index to internal indices in said partition.

        Parameters
        ----------
        axis : {0, 1}
            The axis along which to get the indices (0 - rows, 1 - columns).
        indices : list of int, slice
            A list of global indices to convert.
        are_indices_sorted : bool, default: False
            Flag indicating whether the `indices` sequence is sorted by ascending or not.
            Note: the internal algorithm requires for the `indices` to be sorted, this
            flag is used for optimization in order to not sort already sorted data.
            Be careful when passing ``True`` for this flag, if the data appears to be unsorted
            with the flag set to ``True`` this would lead to undefined behavior.

        Returns
        -------
        dict
            A mapping from partition index to list of internal indices which correspond to `indices` in each
            partition.
        """
    if isinstance(indices, slice) and (indices.step is not None and indices.step != 1):
        indices = range(*indices.indices(len(self.get_axis(axis))))
    if isinstance(indices, slice) or (is_range_like(indices) and indices.step == 1):
        indices = slice(indices.start, indices.stop, indices.step)
        if is_full_grab_slice(indices, sequence_len=len(self.get_axis(axis))):
            return dict(zip(range(self._partitions.shape[axis]), [slice(None)] * self._partitions.shape[axis]))
        if indices.start == indices.stop and indices.start is not None:
            return dict()
        if indices.start is None or indices.start == 0:
            last_part, last_idx = list(self._get_dict_of_block_index(axis, [indices.stop]).items())[0]
            dict_of_slices = dict(zip(range(last_part), [slice(None)] * last_part))
            dict_of_slices.update({last_part: slice(last_idx[0])})
            return dict_of_slices
        elif indices.stop is None or indices.stop >= len(self.get_axis(axis)):
            first_part, first_idx = list(self._get_dict_of_block_index(axis, [indices.start]).items())[0]
            dict_of_slices = dict({first_part: slice(first_idx[0], None)})
            num_partitions = np.size(self._partitions, axis=axis)
            part_list = range(first_part + 1, num_partitions)
            dict_of_slices.update(dict(zip(part_list, [slice(None)] * len(part_list))))
            return dict_of_slices
        else:
            first_part, first_idx = list(self._get_dict_of_block_index(axis, [indices.start]).items())[0]
            last_part, last_idx = list(self._get_dict_of_block_index(axis, [indices.stop]).items())[0]
            if first_part == last_part:
                return dict({first_part: slice(first_idx[0], last_idx[0])})
            elif last_part - first_part == 1:
                return dict({first_part: slice(first_idx[0], None), last_part: slice(None, last_idx[0])})
            else:
                dict_of_slices = dict({first_part: slice(first_idx[0], None)})
                part_list = range(first_part + 1, last_part)
                dict_of_slices.update(dict(zip(part_list, [slice(None)] * len(part_list))))
                dict_of_slices.update({last_part: slice(None, last_idx[0])})
                return dict_of_slices
    if isinstance(indices, list):
        indices = np.array(indices, dtype=np.int64)
    if isinstance(indices, np.ndarray) and indices.size == 0:
        return dict([(0, np.array([], dtype=np.int64))])
    negative_mask = np.less(indices, 0)
    has_negative = np.any(negative_mask)
    if has_negative:
        indices = indices.copy() if isinstance(indices, np.ndarray) else np.array(indices, dtype=np.int64)
        indices[negative_mask] = indices[negative_mask] % len(self.get_axis(axis))
    if has_negative or not are_indices_sorted:
        indices = np.sort(indices)
    if axis == 0:
        bins = np.array(self.row_lengths)
    else:
        bins = np.array(self.column_widths)
    cumulative = np.append(bins[:-1].cumsum(), np.iinfo(bins.dtype).max)

    def internal(block_idx: int, global_index):
        """Transform global index to internal one for given block (identified by its index)."""
        return global_index if not block_idx else np.subtract(global_index, cumulative[min(block_idx, len(cumulative) - 1) - 1])
    partition_ids = np.digitize(indices, cumulative)
    count_for_each_partition = np.array([(partition_ids == i).sum() for i in range(len(cumulative))]).cumsum()
    if count_for_each_partition[0] > 0:
        first_partition_indices = [(0, internal(0, indices[slice(count_for_each_partition[0])]))]
    else:
        first_partition_indices = []
    partition_ids_with_indices = first_partition_indices + [(i, internal(i, indices[slice(count_for_each_partition[i - 1], count_for_each_partition[i])])) for i in range(1, len(count_for_each_partition)) if count_for_each_partition[i] > count_for_each_partition[i - 1]]
    return dict(partition_ids_with_indices)