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
from fsspec.core import url_to_fs
from huggingface_hub import (
from huggingface_hub.hf_api import RepoFile
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
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike
def to_polars(self, batch_size: Optional[int]=None, batched: bool=False, schema_overrides: Optional[dict]=None, rechunk: bool=True) -> Union['pl.DataFrame', Iterator['pl.DataFrame']]:
    """Returns the dataset as a `polars.DataFrame`. Can also return a generator for large datasets.

        Args:
            batched (`bool`):
                Set to `True` to return a generator that yields the dataset as batches
                of `batch_size` rows. Defaults to `False` (returns the whole datasets once).
            batch_size (`int`, *optional*):
                The size (number of rows) of the batches if `batched` is `True`.
                Defaults to `genomicsml.datasets.config.DEFAULT_MAX_BATCH_SIZE`.
            schema_overrides (`dict`, *optional*):
                Support type specification or override of one or more columns; note that
                any dtypes inferred from the schema param will be overridden.
            rechunk (`bool`):
                Make sure that all data is in contiguous memory. Defaults to `True`.
        Returns:
            `polars.DataFrame` or `Iterator[polars.DataFrame]`

        Example:

        ```py
        >>> ds.to_polars()
        ```
        """
    if config.POLARS_AVAILABLE:
        import polars as pl
        if not batched:
            return pl.from_arrow(query_table(table=self._data, key=slice(0, len(self)), indices=self._indices if self._indices is not None else None), schema_overrides=schema_overrides, rechunk=rechunk)
        else:
            batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
            return (pl.from_arrow(query_table(table=self._data, key=slice(offset, offset + batch_size), indices=self._indices if self._indices is not None else None), schema_overrides=schema_overrides, rechunk=rechunk) for offset in range(0, len(self), batch_size))
    else:
        raise ValueError('Polars needs to be installed to be able to return Polars dataframes.')