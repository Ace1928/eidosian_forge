import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import textwrap
import time
import urllib
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
from unittest.mock import patch
import fsspec
import pyarrow as pa
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager, xjoin, xopen
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError, ManualDownloadError
from .features import Features
from .filesystems import (
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable, ExamplesIterable, IterableDataset
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.file_utils import cached_path, is_remote_url
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list
def _relative_data_dir(self, with_version=True, with_hash=True) -> str:
    """Relative path of this dataset in cache_dir:
        Will be:
            self.dataset_name/self.config.version/self.hash/
        or if a repo_id with a namespace has been specified:
            self.namespace___self.dataset_name/self.config.version/self.hash/
        If any of these element is missing or if ``with_version=False`` the corresponding subfolders are dropped.
        """
    if self._legacy_relative_data_dir is not None and with_version and with_hash:
        return self._legacy_relative_data_dir
    namespace = self.repo_id.split('/')[0] if self.repo_id and self.repo_id.count('/') > 0 else None
    builder_data_dir = self.dataset_name if namespace is None else f'{namespace}___{self.dataset_name}'
    builder_data_dir = posixpath.join(builder_data_dir, self.config_id)
    if with_version:
        builder_data_dir = posixpath.join(builder_data_dir, str(self.config.version))
    if with_hash and self.hash and isinstance(self.hash, str):
        builder_data_dir = posixpath.join(builder_data_dir, self.hash)
    return builder_data_dir