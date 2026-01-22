import filecmp
import glob
import importlib
import inspect
import json
import os
import posixpath
import shutil
import signal
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import fsspec
import requests
import yaml
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFileSystem
from . import config
from .arrow_dataset import Dataset
from .builder import BuilderConfig, DatasetBuilder
from .data_files import (
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xbasename, xglob, xjoin
from .exceptions import DataFilesNotFoundError, DatasetNotFoundError
from .features import Features
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict
from .iterable_dataset import IterableDataset
from .metric import Metric
from .naming import camelcase_to_snakecase, snakecase_to_camelcase
from .packaged_modules import (
from .splits import Split
from .utils import _datasets_server
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.file_utils import (
from .utils.hub import hf_hub_url
from .utils.info_utils import VerificationMode, is_small_dataset
from .utils.logging import get_logger
from .utils.metadata import MetadataConfigs
from .utils.py_utils import get_imports
from .utils.version import Version
def _copy_script_and_other_resources_in_importable_dir(name: str, importable_directory_path: str, subdirectory_name: str, original_local_path: str, local_imports: List[Tuple[str, str]], additional_files: List[Tuple[str, str]], download_mode: Optional[Union[DownloadMode, str]]) -> str:
    """Copy a script and its required imports to an importable directory

    Args:
        name (str): name of the resource to load
        importable_directory_path (str): path to the loadable folder in the dynamic modules directory
        subdirectory_name (str): name of the subdirectory in importable_directory_path in which to place the script
        original_local_path (str): local path to the resource script
        local_imports (List[Tuple[str, str]]): list of (destination_filename, import_file_to_copy)
        additional_files (List[Tuple[str, str]]): list of (destination_filename, additional_file_to_copy)
        download_mode (Optional[Union[DownloadMode, str]]): download mode

    Return:
        importable_local_file: path to an importable module with importlib.import_module
    """
    importable_subdirectory = os.path.join(importable_directory_path, subdirectory_name)
    importable_local_file = os.path.join(importable_subdirectory, name + '.py')
    lock_path = importable_directory_path + '.lock'
    with FileLock(lock_path):
        if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(importable_directory_path):
            shutil.rmtree(importable_directory_path)
        os.makedirs(importable_directory_path, exist_ok=True)
        init_file_path = os.path.join(importable_directory_path, '__init__.py')
        if not os.path.exists(init_file_path):
            with open(init_file_path, 'w'):
                pass
        os.makedirs(importable_subdirectory, exist_ok=True)
        init_file_path = os.path.join(importable_subdirectory, '__init__.py')
        if not os.path.exists(init_file_path):
            with open(init_file_path, 'w'):
                pass
        if not os.path.exists(importable_local_file):
            shutil.copyfile(original_local_path, importable_local_file)
        meta_path = os.path.splitext(importable_local_file)[0] + '.json'
        if not os.path.exists(meta_path):
            meta = {'original file path': original_local_path, 'local file path': importable_local_file}
            with open(meta_path, 'w', encoding='utf-8') as meta_file:
                json.dump(meta, meta_file)
        for import_name, import_path in local_imports:
            if os.path.isfile(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name + '.py')
                if not os.path.exists(full_path_local_import):
                    shutil.copyfile(import_path, full_path_local_import)
            elif os.path.isdir(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name)
                if not os.path.exists(full_path_local_import):
                    shutil.copytree(import_path, full_path_local_import)
            else:
                raise ImportError(f'Error with local import at {import_path}')
        for file_name, original_path in additional_files:
            destination_additional_path = os.path.join(importable_subdirectory, file_name)
            if not os.path.exists(destination_additional_path) or not filecmp.cmp(original_path, destination_additional_path):
                shutil.copyfile(original_path, destination_additional_path)
        return importable_local_file