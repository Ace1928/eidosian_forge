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
def _new_dataset_with_indices(self, indices_cache_file_name: Optional[str]=None, indices_buffer: Optional[pa.Buffer]=None, fingerprint: Optional[str]=None) -> 'Dataset':
    """Return a new Dataset obtained by adding indices (provided in indices_cache_file_name or in a buffer) to the
        current Dataset.
        """
    if indices_cache_file_name is None and indices_buffer is None:
        raise ValueError('At least one of indices_cache_file_name or indices_buffer must be provided.')
    if fingerprint is None:
        raise ValueError('please specify a fingerprint for the dataset with indices')
    if indices_cache_file_name is not None:
        indices_table = MemoryMappedTable.from_file(indices_cache_file_name)
    else:
        indices_table = InMemoryTable.from_buffer(indices_buffer)
    return Dataset(self._data, info=self.info.copy(), split=self.split, indices_table=indices_table, fingerprint=fingerprint)