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
@transmit_format
@fingerprint_transform(inplace=False, ignore_kwargs=['indices_cache_file_name'])
def _select_with_indices_mapping(self, indices: Iterable, keep_in_memory: bool=False, indices_cache_file_name: Optional[str]=None, writer_batch_size: Optional[int]=1000, new_fingerprint: Optional[str]=None) -> 'Dataset':
    """Create a new dataset with rows selected following the list/array of indices.
        The new dataset is made by creating a new indices mapping on top of the main arrow table.

        Args:
            indices (sequence, iterable, range, ndarray or Series): List or 1D-array of integer indices for indexing.
            keep_in_memory (`bool`, default `False`): Keep the indices mapping in memory instead of writing it to a cache file.
            indices_cache_file_name (`str`, optional, default `None`): Provide the name of a path for the cache file. It is used to store the
                indices mapping instead of the automatically generated cache file name.
            writer_batch_size (`int`, default `1000`): Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
            new_fingerprint (`str`, optional, default `None`): the new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds._select_with_indices_mapping(range(4))
        Dataset({
            features: ['text', 'label'],
            num_rows: 4
        })
        ```
        """
    if keep_in_memory and indices_cache_file_name is not None:
        raise ValueError('Please use either `keep_in_memory` or `indices_cache_file_name` but not both.')
    if len(self.list_indexes()) > 0:
        raise DatasetTransformationNotAllowedError('Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.')
    if len(self) == 0:
        return self
    if keep_in_memory or indices_cache_file_name is None:
        buf_writer = pa.BufferOutputStream()
        tmp_file = None
        writer = ArrowWriter(stream=buf_writer, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit='indices')
    else:
        buf_writer = None
        logger.info(f'Caching indices mapping at {indices_cache_file_name}')
        tmp_file = tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(indices_cache_file_name), delete=False)
        writer = ArrowWriter(path=tmp_file.name, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit='indices')
    indices = indices if isinstance(indices, list) else list(indices)
    size = len(self)
    if indices:
        _check_valid_indices_value(int(max(indices)), size=size)
        _check_valid_indices_value(int(min(indices)), size=size)
    else:
        return self._select_contiguous(0, 0, new_fingerprint=new_fingerprint)
    indices_array = pa.array(indices, type=pa.uint64())
    if self._indices is not None:
        indices_array = self._indices.column(0).take(indices_array)
    indices_table = pa.Table.from_arrays([indices_array], names=['indices'])
    with writer:
        try:
            writer.write_table(indices_table)
            writer.finalize()
        except (Exception, KeyboardInterrupt):
            if tmp_file is not None:
                tmp_file.close()
                if os.path.exists(tmp_file.name):
                    os.remove(tmp_file.name)
            raise
    if tmp_file is not None:
        tmp_file.close()
        shutil.move(tmp_file.name, indices_cache_file_name)
        umask = os.umask(438)
        os.umask(umask)
        os.chmod(indices_cache_file_name, 438 & ~umask)
    if buf_writer is None:
        return self._new_dataset_with_indices(indices_cache_file_name=indices_cache_file_name, fingerprint=new_fingerprint)
    else:
        return self._new_dataset_with_indices(indices_buffer=buf_writer.getvalue(), fingerprint=new_fingerprint)