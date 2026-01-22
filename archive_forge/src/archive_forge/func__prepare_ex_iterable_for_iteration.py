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
def _prepare_ex_iterable_for_iteration(self) -> _BaseExamplesIterable:
    if self._shuffling:
        ex_iterable = self._ex_iterable.shuffle_data_sources(self._effective_generator())
    else:
        ex_iterable = self._ex_iterable
    if self._distributed:
        rank = self._distributed.rank
        world_size = self._distributed.world_size
        if ex_iterable.n_shards % world_size == 0:
            if self._is_main_process():
                n_shards_per_node = ex_iterable.n_shards // world_size
                plural = 's' if n_shards_per_node > 1 else ''
                logger.info(f'Assigning {n_shards_per_node} shard{plural} (or data source{plural}) of the dataset to each node.')
            ex_iterable = ex_iterable.shard_data_sources(rank, world_size)
        else:
            if self._is_main_process():
                logger.info(f'Assigning 1 out of {world_size} examples of the dataset to each node. The others are skipped during the iteration.')
                logger.info(f'It is more optimized to distribute the dataset shards (or data sources) across nodes. You can do that by using a dataset with number of shards that is a factor of world_size={world_size}. The current dataset has {ex_iterable.n_shards} which is not a factor of {world_size}')
            ex_iterable = StepExamplesIterable(ex_iterable, step=world_size, offset=rank)
    return ex_iterable