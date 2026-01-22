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
@staticmethod
def _get_output_signature(dataset: 'Dataset', collate_fn: Callable, collate_fn_args: dict, cols_to_retain: Optional[List[str]]=None, batch_size: Optional[int]=None, num_test_batches: int=20):
    """Private method used by `to_tf_dataset()` to find the shapes and dtypes of samples from this dataset
           after being passed through the collate_fn. Tensorflow needs an exact signature for tf.numpy_function, so
           the only way to do this is to run test batches - the collator may add or rename columns, so we can't figure
           it out just by inspecting the dataset.

        Args:
            dataset (`Dataset`): Dataset to load samples from.
            collate_fn(`bool`): Shuffle the dataset order when loading. Recommended True for training, False for
                validation/evaluation.
            collate_fn(`Callable`): A function or callable object (such as a `DataCollator`) that will collate
                lists of samples into a batch.
            collate_fn_args (`Dict`): A `dict` of keyword arguments to be passed to the
                `collate_fn`.
            batch_size (`int`, optional): The size of batches loaded from the dataset. Used for shape inference.
                Can be None, which indicates that batch sizes can be variable.
            num_test_batches (`int`): The number of batches to load from the dataset for shape inference.

        Returns:
            `dict`: Dict mapping column names to tf.Tensorspec objects
            `dict`: Dict mapping column names to np.dtype objects
        """
    if config.TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError('Called a Tensorflow-specific function but Tensorflow is not installed.')
    if len(dataset) == 0:
        raise ValueError('Unable to get the output signature because the dataset is empty.')
    if batch_size is not None:
        batch_size = min(len(dataset), batch_size)
    test_batch_size = 1
    if cols_to_retain is not None:
        cols_to_retain = list(set(cols_to_retain + ['label_ids', 'label', 'labels']))
    test_batches = []
    for _ in range(num_test_batches):
        indices = sample(range(len(dataset)), test_batch_size)
        test_batch = dataset[indices]
        if cols_to_retain is not None:
            test_batch = {key: value for key, value in test_batch.items() if key in cols_to_retain}
        test_batch = [{key: value[i] for key, value in test_batch.items()} for i in range(test_batch_size)]
        test_batch = collate_fn(test_batch, **collate_fn_args)
        test_batches.append(test_batch)
    tf_columns_to_signatures = {}
    np_columns_to_dtypes = {}
    for column in test_batches[0].keys():
        raw_arrays = [batch[column] for batch in test_batches]
        np_arrays = []
        for array in raw_arrays:
            if isinstance(array, np.ndarray):
                np_arrays.append(array)
            elif isinstance(array, tf.Tensor):
                np_arrays.append(array.numpy())
            else:
                np_arrays.append(np.array(array))
        if np.issubdtype(np_arrays[0].dtype, np.integer) or np_arrays[0].dtype == bool:
            tf_dtype = tf.int64
            np_dtype = np.int64
        elif np.issubdtype(np_arrays[0].dtype, np.number):
            tf_dtype = tf.float32
            np_dtype = np.float32
        elif np_arrays[0].dtype.kind == 'U':
            np_dtype = np.unicode_
            tf_dtype = tf.string
        else:
            raise RuntimeError(f'Unrecognized array dtype {np_arrays[0].dtype}. \nNested types and image/audio types are not supported yet.')
        shapes = [array.shape for array in np_arrays]
        static_shape = []
        for dim in range(len(shapes[0])):
            sizes = {shape[dim] for shape in shapes}
            if dim == 0:
                static_shape.append(batch_size)
                continue
            if len(sizes) == 1:
                static_shape.append(sizes.pop())
            else:
                static_shape.append(None)
        tf_columns_to_signatures[column] = tf.TensorSpec(shape=static_shape, dtype=tf_dtype)
        np_columns_to_dtypes[column] = np_dtype
    return (tf_columns_to_signatures, np_columns_to_dtypes)