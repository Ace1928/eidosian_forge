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
def _feature(values: Union[float, int, str, np.ndarray, list]) -> 'tf.train.Feature':
    """Typechecks `values` and returns the corresponding tf.train.Feature."""
    if isinstance(values, list):
        if values and isinstance(values[0], str):
            return _bytes_feature([v.encode() for v in values])
        else:
            raise ValueError(f'values={values} is empty or contains items that cannot be serialized')
    elif isinstance(values, np.ndarray):
        if values.dtype == np.dtype(float):
            return _float_feature(values)
        elif values.dtype == np.int64:
            return _int64_feature(values)
        elif values.dtype == np.dtype(str) or (values.dtype == np.dtype(object) and len(values) > 0 and isinstance(values[0], str)):
            return _bytes_feature([v.encode() for v in values])
        else:
            raise ValueError(f'values={values} is empty or is an np.ndarray with items of dtype {values[0].dtype}, which cannot be serialized')
    elif hasattr(values, 'dtype'):
        if np.issubdtype(values.dtype, np.floating):
            return _float_feature([values.item()])
        elif np.issubdtype(values.dtype, np.integer):
            return _int64_feature([values.item()])
        elif np.issubdtype(values.dtype, str):
            return _bytes_feature([values.item().encode()])
        else:
            raise ValueError(f'values={values} has dtype {values.dtype}, which cannot be serialized')
    else:
        raise ValueError(f'values={values} are not numpy objects or strings, and so cannot be serialized')