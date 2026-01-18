import collections
import copy
import html
import itertools
import logging
import time
import warnings
from typing import (
import numpy as np
import ray
import ray.cloudpickle as pickle
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray._private.usage import usage_lib
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.equalize import _equalize
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.legacy_compat import _block_list_to_bundles
from ray.data._internal.iterator.iterator_impl import DataIteratorImpl
from ray.data._internal.iterator.stream_split_iterator import StreamSplitDataIterator
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.map_operator import (
from ray.data._internal.logical.operators.n_ary_operator import (
from ray.data._internal.logical.operators.n_ary_operator import Zip
from ray.data._internal.logical.operators.one_to_one_operator import Limit
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.pandas_block import PandasBlockSchema
from ray.data._internal.plan import ExecutionPlan, OneToOneStage
from ray.data._internal.planner.plan_udf_map_op import (
from ray.data._internal.planner.plan_write_op import generate_write_fn
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data._internal.split import _get_num_rows, _split_at_indices
from ray.data._internal.stage_impl import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary, StatsManager
from ray.data._internal.util import (
from ray.data.aggregate import AggregateFn, Max, Mean, Min, Std, Sum
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import (
from ray.data.iterator import DataIterator
from ray.data.random_access_dataset import RandomAccessDataset
from ray.types import ObjectRef
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
@ConsumptionAPI(pattern='Time complexity:')
def to_torch(self, *, label_column: Optional[str]=None, feature_columns: Optional[Union[List[str], List[List[str]], Dict[str, List[str]]]]=None, label_column_dtype: Optional['torch.dtype']=None, feature_column_dtypes: Optional[Union['torch.dtype', List['torch.dtype'], Dict[str, 'torch.dtype']]]=None, batch_size: int=1, prefetch_batches: int=1, drop_last: bool=False, local_shuffle_buffer_size: Optional[int]=None, local_shuffle_seed: Optional[int]=None, unsqueeze_label_tensor: bool=True, unsqueeze_feature_tensors: bool=True, prefetch_blocks: int=0) -> 'torch.utils.data.IterableDataset':
    """Return a
        `Torch IterableDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`_
        over this :class:`~ray.data.Dataset`.

        This is only supported for datasets convertible to Arrow records.

        It is recommended to use the returned ``IterableDataset`` directly
        instead of passing it into a torch ``DataLoader``.

        Each element in ``IterableDataset`` is a tuple consisting of 2
        elements. The first item contains the feature tensor(s), and the
        second item is the label tensor. Those can take on different
        forms, depending on the specified arguments.

        For the features tensor (N is the ``batch_size`` and n, m, k
        are the number of features per tensor):

        * If ``feature_columns`` is a ``List[str]``, the features is
          a tensor of shape (N, n), with columns corresponding to
          ``feature_columns``

        * If ``feature_columns`` is a ``List[List[str]]``, the features is
          a list of tensors of shape [(N, m),...,(N, k)], with columns of each
          tensor corresponding to the elements of ``feature_columns``

        * If ``feature_columns`` is a ``Dict[str, List[str]]``, the features
          is a dict of key-tensor pairs of shape
          {key1: (N, m),..., keyN: (N, k)}, with columns of each
          tensor corresponding to the value of ``feature_columns`` under the
          key.

        If ``unsqueeze_label_tensor=True`` (default), the label tensor is
        of shape (N, 1). Otherwise, it is of shape (N,).
        If ``label_column`` is specified as ``None``, then no column from the
        ``Dataset`` is treated as the label, and the output label tensor
        is ``None``.

        Note that you probably want to call :meth:`Dataset.split` on this dataset if
        there are to be multiple Torch workers consuming the data.

        Time complexity: O(1)

        Args:
            label_column: The name of the column used as the
                label (second element of the output list). Can be None for
                prediction, in which case the second element of returned
                tuple will also be None.
            feature_columns: The names of the columns
                to use as the features. Can be a list of lists or
                a dict of string-list pairs for multi-tensor output.
                If ``None``, then use all columns except the label column as
                the features.
            label_column_dtype: The torch dtype to
                use for the label column. If ``None``, then automatically infer
                the dtype.
            feature_column_dtypes: The dtypes to use for the feature
                tensors. This should match the format of ``feature_columns``,
                or be a single dtype, in which case it is applied to
                all tensors. If ``None``, then automatically infer the dtype.
            batch_size: How many samples per batch to yield at a time.
                Defaults to 1.
            prefetch_batches: The number of batches to fetch ahead of the current batch
                to fetch. If set to greater than 0, a separate threadpool is used
                to fetch the objects to the local node, format the batches, and apply
                the collate_fn. Defaults to 1.
            drop_last: Set to True to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If
                False and the size of the stream is not divisible by the batch
                size, then the last batch is smaller. Defaults to False.
            local_shuffle_buffer_size: If non-None, the data is randomly shuffled
                using a local in-memory shuffle buffer, and this value will serve as the
                minimum number of rows that must be in the local in-memory shuffle
                buffer in order to yield a batch. When there are no more rows to add to
                the buffer, the remaining rows in the buffer is drained. This
                buffer size must be greater than or equal to ``batch_size``, and
                therefore ``batch_size`` must also be specified when using local
                shuffling.
            local_shuffle_seed: The seed to use for the local random shuffle.
            unsqueeze_label_tensor: If set to True, the label tensor
                is unsqueezed (reshaped to (N, 1)). Otherwise, it will
                be left as is, that is (N, ). In general, regression loss
                functions expect an unsqueezed tensor, while classification
                loss functions expect a squeezed one. Defaults to True.
            unsqueeze_feature_tensors: If set to True, the features tensors
                are unsqueezed (reshaped to (N, 1)) before being concatenated into
                the final features tensor. Otherwise, they are left as is, that is
                (N, ). Defaults to True.

        Returns:
            A `Torch IterableDataset`_.
        """
    return self.iterator().to_torch(label_column=label_column, feature_columns=feature_columns, label_column_dtype=label_column_dtype, feature_column_dtypes=feature_column_dtypes, batch_size=batch_size, prefetch_blocks=prefetch_blocks, prefetch_batches=prefetch_batches, drop_last=drop_last, local_shuffle_buffer_size=local_shuffle_buffer_size, local_shuffle_seed=local_shuffle_seed, unsqueeze_label_tensor=unsqueeze_label_tensor, unsqueeze_feature_tensors=unsqueeze_feature_tensors)