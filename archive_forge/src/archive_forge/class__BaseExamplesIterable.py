import copy
import itertools
import sys
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import cycle, islice
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from . import config
from .arrow_dataset import Dataset, DatasetInfoMixin
from .features import Features
from .features.features import FeatureType, _align_features, _check_if_features_can_be_aligned, cast_to_python_objects
from .filesystems import _reset_fsspec_lock
from .formatting import PythonFormatter, TensorFormatter, get_format_type_from_alias, get_formatter
from .info import DatasetInfo
from .splits import NamedSplit
from .table import cast_table_to_features, read_schema_from_file, table_cast
from .utils.logging import get_logger
from .utils.py_utils import Literal
from .utils.sharding import _merge_gen_kwargs, _number_of_shards_in_gen_kwargs, _shuffle_gen_kwargs, _split_gen_kwargs
class _BaseExamplesIterable:
    """Base class for the examples iterable used by an IterableDataset"""

    def __init__(self) -> None:
        self.iter_arrow: Optional[Callable[[], Iterator[Tuple[Key, pa.Table]]]] = None

    def __iter__(self) -> Iterator[Tuple[Key, dict]]:
        """An examples iterable should yield tuples (example_key, example) of type (int/str, dict)"""
        raise NotImplementedError(f"{type(self)} doesn't implement __iter__ yet")

    def shuffle_data_sources(self, generator: np.random.Generator) -> '_BaseExamplesIterable':
        """
        Either shuffle the shards/sources of the dataset, or propagate the shuffling to the underlying iterable.
        If the order of the shards must stay fixed (when using .skip or .take for example), then this method returns self.
        """
        raise NotImplementedError(f"{type(self)} doesn't implement shuffle_data_sources yet")

    def shard_data_sources(self, worker_id: int, num_workers: int) -> '_BaseExamplesIterable':
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        raise NotImplementedError(f"{type(self)} doesn't implement shard_data_sources yet")

    def split_shard_indices_by_worker(self, worker_id: int, num_workers: int) -> List[int]:
        return list(range(worker_id, self.n_shards, num_workers))

    @property
    def n_shards(self) -> int:
        raise NotImplementedError(f"{type(self)} doesn't implement n_shards yet")