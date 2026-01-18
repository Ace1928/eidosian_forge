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
def write_json(self, path: str, *, filesystem: Optional['pyarrow.fs.FileSystem']=None, try_create_dir: bool=True, arrow_open_stream_args: Optional[Dict[str, Any]]=None, filename_provider: Optional[FilenameProvider]=None, block_path_provider: Optional[BlockWritePathProvider]=None, pandas_json_args_fn: Callable[[], Dict[str, Any]]=lambda: {}, ray_remote_args: Dict[str, Any]=None, **pandas_json_args) -> None:
    """Writes the :class:`~ray.data.Dataset` to JSON and JSONL files.

        The number of files is determined by the number of blocks in the dataset.
        To control the number of number of blocks, call
        :meth:`~ray.data.Dataset.repartition`.

        This method is only supported for datasets with records that are convertible to
        pandas dataframes.

        By default, the format of the output files is ``{uuid}_{block_idx}.json``,
        where ``uuid`` is a unique id for the dataset. To modify this behavior,
        implement a custom
        :class:`~ray.data.file_based_datasource.BlockWritePathProvider`
        and pass it in as the ``block_path_provider`` argument.

        Examples:
            Write the dataset as JSON file to a local directory.

            >>> import ray
            >>> import pandas as pd
            >>> ds = ray.data.from_pandas([pd.DataFrame({"one": [1], "two": ["a"]})])
            >>> ds.write_json("local:///tmp/data")

            Write the dataset as JSONL files to a local directory.

            >>> ds = ray.data.read_json("s3://anonymous@ray-example-data/train.jsonl")
            >>> ds.write_json("local:///tmp/data")

        Time complexity: O(dataset size / parallelism)

        Args:
            path: The path to the destination root directory, where
                the JSON files are written to.
            filesystem: The pyarrow filesystem implementation to write to.
                These filesystems are specified in the
                `pyarrow docs <https://arrow.apache.org/docs                /python/api/filesystems.html#filesystem-implementations>`_.
                Specify this if you need to provide specific configurations to the
                filesystem. By default, the filesystem is automatically selected based
                on the scheme of the paths. For example, if the path begins with
                ``s3://``, the ``S3FileSystem`` is used.
            try_create_dir: If ``True``, attempts to create all directories in the
                destination path. Does nothing if all directories already
                exist. Defaults to ``True``.
            arrow_open_stream_args: kwargs passed to
                `pyarrow.fs.FileSystem.open_output_stream <https://arrow.apache.org                /docs/python/generated/pyarrow.fs.FileSystem.html                #pyarrow.fs.FileSystem.open_output_stream>`_, which is used when
                opening the file to write to.
            filename_provider: A :class:`~ray.data.datasource.FilenameProvider`
                implementation. Use this parameter to customize what your filenames
                look like.
            pandas_json_args_fn: Callable that returns a dictionary of write
                arguments that are provided to
                `pandas.DataFrame.to_json() <https://pandas.pydata.org/docs/reference/                    api/pandas.DataFrame.to_json.html>`_
                when writing each block to a file. Overrides
                any duplicate keys from ``pandas_json_args``. Use this parameter
                instead of ``pandas_json_args`` if any of your write arguments
                can't be pickled, or if you'd like to lazily resolve the write
                arguments for each dataset block.
            ray_remote_args: kwargs passed to :meth:`~ray.remote` in the write tasks.
            pandas_json_args: These args are passed to
                `pandas.DataFrame.to_json() <https://pandas.pydata.org/docs/reference/                    api/pandas.DataFrame.to_json.html>`_,
                which is used under the hood to write out each
                :class:`~ray.data.Dataset` block. These
                are dict(orient="records", lines=True) by default.
        """
    datasink = _JSONDatasink(path, pandas_json_args_fn=pandas_json_args_fn, pandas_json_args=pandas_json_args, filesystem=filesystem, try_create_dir=try_create_dir, open_stream_args=arrow_open_stream_args, filename_provider=filename_provider, block_path_provider=block_path_provider, dataset_uuid=self._uuid)
    self.write_datasink(datasink, ray_remote_args=ray_remote_args)