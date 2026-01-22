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
def _build_cache_dir(self):
    """Return the data directory for the current version."""
    builder_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=False))
    version_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=True))

    def _other_versions_on_disk():
        """Returns previous versions on disk."""
        if not os.path.exists(builder_data_dir):
            return []
        version_dirnames = []
        for dir_name in os.listdir(builder_data_dir):
            try:
                version_dirnames.append((utils.Version(dir_name), dir_name))
            except ValueError:
                pass
        version_dirnames.sort(reverse=True)
        return version_dirnames
    if not is_remote_url(builder_data_dir):
        version_dirs = _other_versions_on_disk()
        if version_dirs:
            other_version = version_dirs[0][0]
            if other_version != self.config.version:
                warn_msg = f'Found a different version {str(other_version)} of dataset {self.dataset_name} in cache_dir {self._cache_dir_root}. Using currently defined version {str(self.config.version)}.'
                logger.warning(warn_msg)
    return version_data_dir