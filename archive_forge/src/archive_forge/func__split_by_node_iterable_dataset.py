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
def _split_by_node_iterable_dataset(dataset: IterableDataset, rank: int, world_size: int) -> IterableDataset:
    """
    Split an iterable dataset for the node at rank `rank` in a pool of nodes of size `world_size`.

    If the dataset has a number of shards that is a factor of `world_size` (i.e. if `dataset.n_shards % world_size == 0`),
    then the shards are evenly assigned across the nodes, which is the most optimized.
    Otherwise, each node keeps 1 example out of `world_size`, skipping the other examples.

    Args:
        dataset ([`IterableDataset`]):
            The iterable dataset to split by node.
        rank (`int`):
            Rank of the current node.
        world_size (`int`):
            Total number of nodes.

    Returns:
        [`IterableDataset`]: The iterable dataset to be used on the node at rank `rank`.
    """
    if dataset._distributed:
        world_size = world_size * dataset._distributed.world_size
        rank = world_size * dataset._distributed.rank + rank
    distributed = DistributedConfig(rank=rank, world_size=world_size)
    return IterableDataset(ex_iterable=dataset._ex_iterable, info=dataset._info.copy(), split=dataset._split, formatting=dataset._formatting, shuffling=copy.deepcopy(dataset._shuffling), distributed=distributed, token_per_repo_id=dataset._token_per_repo_id)