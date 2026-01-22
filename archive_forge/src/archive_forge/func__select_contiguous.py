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
@fingerprint_transform(inplace=False)
def _select_contiguous(self, start: int, length: int, new_fingerprint: Optional[str]=None) -> 'Dataset':
    """Create a new dataset with rows from a contiguous slice of data.
        The slice is defined by that start index and its length.

        Args:
            start (`int`): start index.
            length (`int`): length of the slice to select.
            new_fingerprint (`str`, optional, default `None`): the new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds._select_contiguous(0, 4)
        Dataset({
            features: ['text', 'label'],
            num_rows: 4
        })
        ```
        """
    if len(self.list_indexes()) > 0:
        raise DatasetTransformationNotAllowedError('Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.')
    if len(self) == 0:
        return self
    _check_valid_indices_value(start, len(self))
    _check_valid_indices_value(start + length - 1, len(self))
    if self._indices is None or length == 0:
        return Dataset(self.data.slice(start, length), info=self.info.copy(), split=self.split, fingerprint=new_fingerprint)
    else:
        return Dataset(self.data, info=self.info.copy(), split=self.split, indices_table=self._indices.slice(start, length), fingerprint=new_fingerprint)