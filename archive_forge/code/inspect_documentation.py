import inspect
import os
import shutil
import warnings
from pathlib import Path, PurePath
from typing import Dict, List, Mapping, Optional, Sequence, Union
import huggingface_hub
from . import config
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager
from .info import DatasetInfo
from .load import (
from .utils.deprecation_utils import deprecated
from .utils.file_utils import relative_to_absolute_path
from .utils.logging import get_logger
from .utils.version import Version
Get the list of available splits for a particular config and dataset.

    Args:
        path (`str`): path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`datasets.list_datasets`])
                e.g. `'squad'`, `'glue'` or `'openai/webtext'`
        config_name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        use_auth_token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>

        **config_kwargs (additional keyword arguments):
            Optional attributes for builder class which will override the attributes if supplied.

    Example:

    ```py
    >>> from datasets import get_dataset_split_names
    >>> get_dataset_split_names('rotten_tomatoes')
    ['train', 'validation', 'test']
    ```
    