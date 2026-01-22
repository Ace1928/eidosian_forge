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
class HubDatasetModuleFactoryWithScript(_DatasetModuleFactory):
    """
    Get the module of a dataset from a dataset repository.
    The dataset script comes from the script inside the dataset repository.
    """

    def __init__(self, name: str, revision: Optional[Union[str, Version]]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, dynamic_modules_path: Optional[str]=None, trust_remote_code: Optional[bool]=None):
        self.name = name
        self.revision = revision
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        self.trust_remote_code = trust_remote_code
        increase_load_count(name, resource_type='dataset')

    def download_loading_script(self) -> str:
        file_path = hf_hub_url(self.name, self.name.split('/')[-1] + '.py', revision=self.revision)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = 'Downloading builder script'
        return cached_path(file_path, download_config=download_config)

    def download_dataset_infos_file(self) -> str:
        dataset_infos = hf_hub_url(self.name, config.DATASETDICT_INFOS_FILENAME, revision=self.revision)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = 'Downloading metadata'
        try:
            return cached_path(dataset_infos, download_config=download_config)
        except (FileNotFoundError, ConnectionError):
            return None

    def download_dataset_readme_file(self) -> str:
        readme_url = hf_hub_url(self.name, config.REPOCARD_FILENAME, revision=self.revision)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = 'Downloading readme'
        try:
            return cached_path(readme_url, download_config=download_config)
        except (FileNotFoundError, ConnectionError):
            return None

    def get_module(self) -> DatasetModule:
        if config.HF_DATASETS_TRUST_REMOTE_CODE and self.trust_remote_code is None:
            warnings.warn(f'The repository for {self.name} contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/{self.name}\nYou can avoid this message in future by passing the argument `trust_remote_code=True`.\nPassing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.', FutureWarning)
        local_path = self.download_loading_script()
        dataset_infos_path = self.download_dataset_infos_file()
        dataset_readme_path = self.download_dataset_readme_file()
        imports = get_imports(local_path)
        local_imports = _download_additional_modules(name=self.name, base_path=hf_hub_url(self.name, '', revision=self.revision), imports=imports, download_config=self.download_config)
        additional_files = []
        if dataset_infos_path:
            additional_files.append((config.DATASETDICT_INFOS_FILENAME, dataset_infos_path))
        if dataset_readme_path:
            additional_files.append((config.REPOCARD_FILENAME, dataset_readme_path))
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        hash = files_to_hash([local_path] + [loc[1] for loc in local_imports])
        importable_file_path = _get_importable_file_path(dynamic_modules_path=dynamic_modules_path, module_namespace='datasets', subdirectory_name=hash, name=self.name)
        if not os.path.exists(importable_file_path):
            trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
            if trust_remote_code:
                _create_importable_file(local_path=local_path, local_imports=local_imports, additional_files=additional_files, dynamic_modules_path=dynamic_modules_path, module_namespace='datasets', subdirectory_name=hash, name=self.name, download_mode=self.download_mode)
            else:
                raise ValueError(f'Loading {self.name} requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.')
        module_path, hash = _load_importable_file(dynamic_modules_path=dynamic_modules_path, module_namespace='datasets', subdirectory_name=hash, name=self.name)
        importlib.invalidate_caches()
        builder_kwargs = {'base_path': hf_hub_url(self.name, '', revision=self.revision).rstrip('/'), 'repo_id': self.name}
        return DatasetModule(module_path, hash, builder_kwargs)