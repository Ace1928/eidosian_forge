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
def _download_additional_modules(name: str, base_path: str, imports: Tuple[str, str, str, str], download_config: Optional[DownloadConfig]) -> List[Tuple[str, str]]:
    """
    Download additional module for a module <name>.py at URL (or local path) <base_path>/<name>.py
    The imports must have been parsed first using ``get_imports``.

    If some modules need to be installed with pip, an error is raised showing how to install them.
    This function return the list of downloaded modules as tuples (import_name, module_file_path).

    The downloaded modules can then be moved into an importable directory with ``_copy_script_and_other_resources_in_importable_dir``.
    """
    local_imports = []
    library_imports = []
    download_config = download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = 'Downloading extra modules'
    for import_type, import_name, import_path, sub_directory in imports:
        if import_type == 'library':
            library_imports.append((import_name, import_path))
            continue
        if import_name == name:
            raise ValueError(f"Error in the {name} script, importing relative {import_name} module but {import_name} is the name of the script. Please change relative import {import_name} to another name and add a '# From: URL_OR_PATH' comment pointing to the original relative import file path.")
        if import_type == 'internal':
            url_or_filename = url_or_path_join(base_path, import_path + '.py')
        elif import_type == 'external':
            url_or_filename = import_path
        else:
            raise ValueError('Wrong import_type')
        local_import_path = cached_path(url_or_filename, download_config=download_config)
        if sub_directory is not None:
            local_import_path = os.path.join(local_import_path, sub_directory)
        local_imports.append((import_name, local_import_path))
    needs_to_be_installed = {}
    for library_import_name, library_import_path in library_imports:
        try:
            lib = importlib.import_module(library_import_name)
        except ImportError:
            if library_import_name not in needs_to_be_installed or library_import_path != library_import_name:
                needs_to_be_installed[library_import_name] = library_import_path
    if needs_to_be_installed:
        _dependencies_str = 'dependencies' if len(needs_to_be_installed) > 1 else 'dependency'
        _them_str = 'them' if len(needs_to_be_installed) > 1 else 'it'
        if 'sklearn' in needs_to_be_installed.keys():
            needs_to_be_installed['sklearn'] = 'scikit-learn'
        if 'Bio' in needs_to_be_installed.keys():
            needs_to_be_installed['Bio'] = 'biopython'
        raise ImportError(f"To be able to use {name}, you need to install the following {_dependencies_str}: {', '.join(needs_to_be_installed)}.\nPlease install {_them_str} using 'pip install {' '.join(needs_to_be_installed.values())}' for instance.")
    return local_imports