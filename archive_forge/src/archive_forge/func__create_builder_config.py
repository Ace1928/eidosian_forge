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
def _create_builder_config(self, config_name=None, custom_features=None, **config_kwargs) -> Tuple[BuilderConfig, str]:
    """Create and validate BuilderConfig object as well as a unique config id for this config.
        Raises ValueError if there are multiple builder configs and config_name and DEFAULT_CONFIG_NAME are None.
        config_kwargs override the defaults kwargs in config
        """
    builder_config = None
    if config_name is None and self.BUILDER_CONFIGS:
        if self.DEFAULT_CONFIG_NAME is not None:
            builder_config = self.builder_configs.get(self.DEFAULT_CONFIG_NAME)
            logger.info(f'No config specified, defaulting to: {self.dataset_name}/{builder_config.name}')
        elif len(self.BUILDER_CONFIGS) > 1:
            if not config_kwargs:
                example_of_usage = f"load_dataset('{self.dataset_name}', '{self.BUILDER_CONFIGS[0].name}')"
                raise ValueError(f'Config name is missing.\nPlease pick one among the available configs: {list(self.builder_configs.keys())}' + f'\nExample of usage:\n\t`{example_of_usage}`')
        else:
            builder_config = self.BUILDER_CONFIGS[0]
            logger.info(f'No config specified, defaulting to the single config: {self.dataset_name}/{builder_config.name}')
    if isinstance(config_name, str):
        builder_config = self.builder_configs.get(config_name)
        if builder_config is None and self.BUILDER_CONFIGS:
            raise ValueError(f"BuilderConfig '{config_name}' not found. Available: {list(self.builder_configs.keys())}")
    if not builder_config:
        if config_name is not None:
            config_kwargs['name'] = config_name
        elif self.DEFAULT_CONFIG_NAME and (not config_kwargs):
            config_kwargs['name'] = self.DEFAULT_CONFIG_NAME
        if 'version' not in config_kwargs and hasattr(self, 'VERSION') and self.VERSION:
            config_kwargs['version'] = self.VERSION
        builder_config = self.BUILDER_CONFIG_CLASS(**config_kwargs)
    else:
        builder_config = copy.deepcopy(builder_config) if config_kwargs else builder_config
        for key, value in config_kwargs.items():
            if value is not None:
                if not hasattr(builder_config, key):
                    raise ValueError(f"BuilderConfig {builder_config} doesn't have a '{key}' key.")
                setattr(builder_config, key, value)
    if not builder_config.name:
        raise ValueError(f'BuilderConfig must have a name, got {builder_config.name}')
    builder_config._resolve_data_files(base_path=self.base_path, download_config=DownloadConfig(token=self.token, storage_options=self.storage_options))
    config_id = builder_config.create_config_id(config_kwargs, custom_features=custom_features)
    is_custom = config_id not in self.builder_configs and config_id != 'default'
    if is_custom:
        logger.info(f'Using custom data configuration {config_id}')
    else:
        if builder_config.name in self.builder_configs and builder_config != self.builder_configs[builder_config.name]:
            raise ValueError(f'Cannot name a custom BuilderConfig the same as an available BuilderConfig. Change the name. Available BuilderConfigs: {list(self.builder_configs.keys())}')
        if not builder_config.version:
            raise ValueError(f'BuilderConfig {builder_config.name} must have a version')
    return (builder_config, config_id)