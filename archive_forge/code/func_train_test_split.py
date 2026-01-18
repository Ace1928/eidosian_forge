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
@ConsumptionAPI
def train_test_split(self, test_size: Union[int, float], *, shuffle: bool=False, seed: Optional[int]=None) -> Tuple['MaterializedDataset', 'MaterializedDataset']:
    """Materialize and split the dataset into train and test subsets.

        Examples:

            >>> import ray
            >>> ds = ray.data.range(8)
            >>> train, test = ds.train_test_split(test_size=0.25)
            >>> train.take_batch()
            {'id': array([0, 1, 2, 3, 4, 5])}
            >>> test.take_batch()
            {'id': array([6, 7])}

        Args:
            test_size: If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the test split. If int,
                represents the absolute number of test samples. The train split
                always complements the test split.
            shuffle: Whether or not to globally shuffle the dataset before splitting.
                Defaults to ``False``. This may be a very expensive operation with a
                large dataset.
            seed: Fix the random seed to use for shuffle, otherwise one is chosen
                based on system randomness. Ignored if ``shuffle=False``.

        Returns:
            Train and test subsets as two ``MaterializedDatasets``.

        .. seealso::

            :meth:`Dataset.split_proportionately`
        """
    ds = self
    if shuffle:
        ds = ds.random_shuffle(seed=seed)
    if not isinstance(test_size, (int, float)):
        raise TypeError(f'`test_size` must be int or float got {type(test_size)}.')
    if isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError(f'If `test_size` is a float, it must be bigger than 0 and smaller than 1. Got {test_size}.')
        return ds.split_proportionately([1 - test_size])
    else:
        ds_length = ds.count()
        if test_size <= 0 or test_size >= ds_length:
            raise ValueError(f'If `test_size` is an int, it must be bigger than 0 and smaller than the size of the dataset ({ds_length}). Got {test_size}.')
        return ds.split_at_indices([ds_length - test_size])