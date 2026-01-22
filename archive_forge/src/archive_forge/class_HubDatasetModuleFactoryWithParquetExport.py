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
class HubDatasetModuleFactoryWithParquetExport(_DatasetModuleFactory):
    """
    Get the module of a dataset loaded from parquet files of a dataset repository parquet export.
    """

    def __init__(self, name: str, revision: Optional[str]=None, download_config: Optional[DownloadConfig]=None):
        self.name = name
        self.revision = revision
        self.download_config = download_config or DownloadConfig()
        increase_load_count(name, resource_type='dataset')

    def get_module(self) -> DatasetModule:
        exported_parquet_files = _datasets_server.get_exported_parquet_files(dataset=self.name, revision=self.revision, token=self.download_config.token)
        exported_dataset_infos = _datasets_server.get_exported_dataset_infos(dataset=self.name, revision=self.revision, token=self.download_config.token)
        dataset_infos = DatasetInfosDict({config_name: DatasetInfo.from_dict(exported_dataset_infos[config_name]) for config_name in exported_dataset_infos})
        hfh_dataset_info = HfApi(config.HF_ENDPOINT).dataset_info(self.name, revision='refs/convert/parquet', token=self.download_config.token, timeout=100.0)
        revision = hfh_dataset_info.sha
        metadata_configs = MetadataConfigs._from_exported_parquet_files_and_dataset_infos(revision=revision, exported_parquet_files=exported_parquet_files, dataset_infos=dataset_infos)
        module_path, _ = _PACKAGED_DATASETS_MODULES['parquet']
        builder_configs, default_config_name = create_builder_configs_from_metadata_configs(module_path, metadata_configs, supports_metadata=False, download_config=self.download_config)
        hash = self.revision
        builder_kwargs = {'repo_id': self.name, 'dataset_name': camelcase_to_snakecase(Path(self.name).name)}
        return DatasetModule(module_path, hash, builder_kwargs, dataset_infos=dataset_infos, builder_configs_parameters=BuilderConfigsParameters(metadata_configs=metadata_configs, builder_configs=builder_configs, default_config_name=default_config_name))