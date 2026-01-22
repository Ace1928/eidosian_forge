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
def apply_function_on_filtered_inputs(pa_inputs, indices, check_same_num_examples=False, offset=0):
    """Utility to apply the function on a selection of columns."""
    nonlocal update_data
    inputs = format_table(pa_inputs, 0 if not batched else range(pa_inputs.num_rows), format_columns=input_columns, formatter=input_formatter)
    fn_args = [inputs] if input_columns is None else [inputs[col] for col in input_columns]
    if offset == 0:
        effective_indices = indices
    else:
        effective_indices = [i + offset for i in indices] if isinstance(indices, list) else indices + offset
    additional_args = ()
    if with_indices:
        additional_args += (effective_indices,)
    if with_rank:
        additional_args += (rank,)
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
    if isinstance(processed_inputs, LazyDict):
        processed_inputs = {k: v for k, v in processed_inputs.data.items() if k not in processed_inputs.keys_to_format}
        returned_lazy_dict = True
    else:
        returned_lazy_dict = False
    if update_data is None:
        update_data = isinstance(processed_inputs, (Mapping, pa.Table, pd.DataFrame))
        validate_function_output(processed_inputs, indices)
    if not update_data:
        return None
    if shard._format_type or input_columns:
        inputs_to_merge = dict(zip(pa_inputs.column_names, pa_inputs.itercolumns()))
    elif isinstance(inputs, LazyDict):
        inputs_to_merge = {k: v if k not in inputs.keys_to_format else pa_inputs[k] for k, v in inputs.data.items()}
    else:
        inputs_to_merge = inputs
    if remove_columns is not None:
        for column in remove_columns:
            if column in inputs_to_merge:
                inputs_to_merge.pop(column)
            if returned_lazy_dict and column in processed_inputs:
                processed_inputs.pop(column)
    if check_same_num_examples:
        input_num_examples = len(pa_inputs)
        processed_inputs_num_examples = len(processed_inputs[next(iter(processed_inputs.keys()))])
        if input_num_examples != processed_inputs_num_examples:
            raise NumExamplesMismatchError()
    if isinstance(inputs, Mapping) and isinstance(processed_inputs, Mapping):
        return {**inputs_to_merge, **processed_inputs}
    else:
        return processed_inputs