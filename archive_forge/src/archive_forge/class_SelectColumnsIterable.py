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
class SelectColumnsIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterable: _BaseExamplesIterable, column_names: List[str]):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.column_names = column_names
        if self.ex_iterable.iter_arrow:
            self.iter_arrow = self._iter_arrow

    def __iter__(self):
        for idx, row in self.ex_iterable:
            yield (idx, {c: row[c] for c in self.column_names})

    def _iter_arrow(self) -> Iterator[Tuple[Key, pa.Table]]:
        for idx, pa_table in self.ex_iterable.iter_arrow():
            yield (idx, pa_table.select(self.column_names))

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'SelectColumnsIterable':
        return SelectColumnsIterable(self.ex_iterable.shuffle_data_sources(generator), self.column_names)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'SelectColumnsIterable':
        return SelectColumnsIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), self.column_names)

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards