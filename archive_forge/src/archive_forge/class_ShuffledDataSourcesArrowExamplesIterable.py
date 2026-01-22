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
class ShuffledDataSourcesArrowExamplesIterable(ArrowExamplesIterable):

    def __init__(self, generate_tables_fn: Callable[..., Tuple[Key, pa.Table]], kwargs: dict, generator: np.random.Generator):
        super().__init__(generate_tables_fn, kwargs)
        self.generator = deepcopy(generator)

    def __iter__(self):
        """Shuffle the kwargs order to shuffle shards"""
        rng = deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        formatter = PythonFormatter()
        for key, pa_table in self.generate_tables_fn(**kwargs_with_shuffled_shards):
            for pa_subtable in pa_table.to_reader(max_chunksize=config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER):
                formatted_batch = formatter.format_batch(pa_subtable)
                for example in _batch_to_examples(formatted_batch):
                    yield (key, example)

    def _iter_arrow(self):
        rng = deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        yield from self.generate_tables_fn(**kwargs_with_shuffled_shards)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'ArrowExamplesIterable':
        """Keep only the requested shard."""
        rng = deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        return ArrowExamplesIterable(self.generate_tables_fn, kwargs_with_shuffled_shards).shard_data_sources(worker_id, num_workers)