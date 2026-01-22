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
class TakeExamplesIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterable: _BaseExamplesIterable, n: int):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.n = n

    def __iter__(self):
        yield from islice(self.ex_iterable, self.n)

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'TakeExamplesIterable':
        """Doesn't shuffle the wrapped examples iterable since it would take examples from other shards instead."""
        return self

    @staticmethod
    def split_number(num, n):
        quotient = num // n
        remainder = num % n
        result = [quotient] * n
        for i in range(remainder):
            result[i] += 1
        return result

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'TakeExamplesIterable':
        """Keep only the requested shard."""
        return TakeExamplesIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), n=self.split_number(self.n, num_workers)[worker_id])

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards