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
def transmit_format(func):
    """Wrapper for dataset transforms that recreate a new Dataset to transmit the format of the original dataset to the new dataset"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: 'Dataset' = args[0]
            args = args[1:]
        else:
            self: 'Dataset' = kwargs.pop('self')
        unformatted_columns = set(self.column_names) - set(self._format_columns or [])
        self_format = {'type': self._format_type, 'format_kwargs': self._format_kwargs, 'columns': self._format_columns, 'output_all_columns': self._output_all_columns}
        out: Union['Dataset', 'DatasetDict'] = func(self, *args, **kwargs)
        datasets: List['Dataset'] = list(out.values()) if isinstance(out, dict) else [out]
        for dataset in datasets:
            new_format = self_format.copy()
            if new_format['columns'] is not None:
                new_format['columns'] = sorted(set(dataset.column_names) - unformatted_columns)
            out_format = {'type': dataset._format_type, 'format_kwargs': dataset._format_kwargs, 'columns': sorted(dataset._format_columns) if dataset._format_columns is not None else None, 'output_all_columns': dataset._output_all_columns}
            if out_format != new_format:
                fingerprint = dataset._fingerprint
                dataset.set_format(**new_format)
                dataset._fingerprint = fingerprint
        return out
    wrapper._decorator_name_ = 'transmit_format'
    return wrapper