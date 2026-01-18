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
def write_bigquery(self, project_id: str, dataset: str, max_retry_cnt: int=10, ray_remote_args: Dict[str, Any]=None) -> None:
    """Write the dataset to a BigQuery dataset table.

        To control the number of parallel write tasks, use ``.repartition()``
        before calling this method.

        Examples:
             .. testcode::
                :skipif: True

                import ray
                import pandas as pd

                docs = [{"title": "BigQuery Datasource test"} for key in range(4)]
                ds = ray.data.from_pandas(pd.DataFrame(docs))
                ds.write_bigquery(
                    project_id="my_project_id",
                    dataset="my_dataset_table",
                )

        Args:
            project_id: The name of the associated Google Cloud Project that hosts
                the dataset to read. For more information, see details in
                `Creating and managing projects <https://cloud.google.com/resource-manager/docs/creating-managing-projects>`.
            dataset: The name of the dataset in the format of ``dataset_id.table_id``.
                The dataset is created if it doesn't already exist. The table_id is
                overwritten if it exists.
            max_retry_cnt: The maximum number of retries that an individual block write
                is retried due to BigQuery rate limiting errors. This isn't
                related to Ray fault tolerance retries. The default number of retries
                is 10.
            ray_remote_args: Kwargs passed to ray.remote in the write tasks.
        """
    if ray_remote_args is None:
        ray_remote_args = {}
    if ray_remote_args.get('max_retries', 0) != 0:
        warnings.warn('The max_retries of a BigQuery Write Task should be set to 0 to avoid duplicate writes.')
    else:
        ray_remote_args['max_retries'] = 0
    datasink = _BigQueryDatasink(project_id, dataset, max_retry_cnt=max_retry_cnt)
    self.write_datasink(datasink, ray_remote_args=ray_remote_args)