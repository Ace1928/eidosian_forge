import contextlib
import copy
import fnmatch
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial, wraps
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from random import sample
from typing import (
from typing import Sequence as Sequence_
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from huggingface_hub import CommitInfo, CommitOperationAdd, CommitOperationDelete, DatasetCard, DatasetCardData, HfApi
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter, OptimizedTypedSequence
from .data_files import sanitize_patterns
from .download.streaming_download_manager import xgetsize
from .features import Audio, ClassLabel, Features, Image, Sequence, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .fingerprint import (
from .formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from .formatting.formatting import LazyDict, _is_range_contiguous
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .search import IndexableMixin
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import (
from .tasks import TaskTemplate
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import estimate_dataset_size
from .utils.hub import list_files_info, preupload_lfs_files
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike
def to_iterable_dataset(self, num_shards: Optional[int]=1) -> 'IterableDataset':
    """Get an [`datasets.IterableDataset`] from a map-style [`datasets.Dataset`].
        This is equivalent to loading a dataset in streaming mode with [`datasets.load_dataset`], but much faster since the data is streamed from local files.

        Contrary to map-style datasets, iterable datasets are lazy and can only be iterated over (e.g. using a for loop).
        Since they are read sequentially in training loops, iterable datasets are much faster than map-style datasets.
        All the transformations applied to iterable datasets like filtering or processing are done on-the-fly when you start iterating over the dataset.

        Still, it is possible to shuffle an iterable dataset using [`datasets.IterableDataset.shuffle`].
        This is a fast approximate shuffling that works best if you have multiple shards and if you specify a buffer size that is big enough.

        To get the best speed performance, make sure your dataset doesn't have an indices mapping.
        If this is the case, the data are not read contiguously, which can be slow sometimes.
        You can use `ds = ds.flatten_indices()` to write your dataset in contiguous chunks of data and have optimal speed before switching to an iterable dataset.

        Args:
            num_shards (`int`, default to `1`):
                Number of shards to define when instantiating the iterable dataset. This is especially useful for big datasets to be able to shuffle properly,
                and also to enable fast parallel loading using a PyTorch DataLoader or in distributed setups for example.
                Shards are defined using [`datasets.Dataset.shard`]: it simply slices the data without writing anything on disk.

        Returns:
            [`datasets.IterableDataset`]

        Example:

        Basic usage:
        ```python
        >>> ids = ds.to_iterable_dataset()
        >>> for example in ids:
        ...     pass
        ```

        With lazy filtering and processing:
        ```python
        >>> ids = ds.to_iterable_dataset()
        >>> ids = ids.filter(filter_fn).map(process_fn)  # will filter and process on-the-fly when you start iterating over the iterable dataset
        >>> for example in ids:
        ...     pass
        ```

        With sharding to enable efficient shuffling:
        ```python
        >>> ids = ds.to_iterable_dataset(num_shards=64)  # the dataset is split into 64 shards to be iterated over
        >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer for fast approximate shuffling when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        With a PyTorch DataLoader:
        ```python
        >>> import torch
        >>> ids = ds.to_iterable_dataset(num_shards=64)
        >>> ids = ids.filter(filter_fn).map(process_fn)
        >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards to each worker to load, filter and process when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        With a PyTorch DataLoader and shuffling:
        ```python
        >>> import torch
        >>> ids = ds.to_iterable_dataset(num_shards=64)
        >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards from the shuffled list of shards to each worker when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        In a distributed setup like PyTorch DDP with a PyTorch DataLoader and shuffling
        ```python
        >>> from datasets.distributed import split_dataset_by_node
        >>> ids = ds.to_iterable_dataset(num_shards=512)
        >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> ids = split_dataset_by_node(ds, world_size=8, rank=0)  # will keep only 512 / 8 = 64 shards from the shuffled lists of shards when you start iterating
        >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards from this node's list of shards to each worker when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        With shuffling and multiple epochs:
        ```python
        >>> ids = ds.to_iterable_dataset(num_shards=64)
        >>> ids = ids.shuffle(buffer_size=10_000, seed=42)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> for epoch in range(n_epochs):
        ...     ids.set_epoch(epoch)  # will use effective_seed = seed + epoch to shuffle the shards and for the shuffle buffer when you start iterating
        ...     for example in ids:
        ...         pass
        ```
        Feel free to also use [`IterableDataset.set_epoch`] when using a PyTorch DataLoader or in distributed setups.
        """
    from .iterable_dataset import ArrowExamplesIterable, IterableDataset
    if self._format_type is not None:
        raise NotImplementedError('Converting a formatted dataset to a formatted iterable dataset is not implemented yet. Please run `my_dataset = my_dataset.with_format(None)` before calling to_iterable_dataset')
    if num_shards > len(self):
        raise ValueError(f'Unable to shard a dataset of size {len(self)} into {num_shards} shards (the number of shards exceeds the number of samples).')
    if self._indices is not None:
        logger.info('Converting an Arrow dataset to iterable but it has an indices mapping that can make it slower. You can use `ds = ds.flatten_indices()` to write your dataset in contiguous chunks of data and have optimal speed.')
    shards = [copy.deepcopy(self)] if num_shards == 1 else [self.shard(num_shards=num_shards, index=shard_idx, contiguous=True) for shard_idx in range(num_shards)]
    ex_iterable = ArrowExamplesIterable(Dataset._generate_tables_from_shards, kwargs={'shards': shards, 'batch_size': config.DEFAULT_MAX_BATCH_SIZE})
    return IterableDataset(ex_iterable, info=DatasetInfo(features=self.features))