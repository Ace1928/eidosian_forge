import os
import re
from functools import partial
from glob import has_magic
from pathlib import Path, PurePath
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import huggingface_hub
from fsspec import get_fs_token_paths
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import HfFileSystem
from packaging import version
from tqdm.contrib.concurrent import thread_map
from . import config
from .download import DownloadConfig
from .download.streaming_download_manager import _prepare_path_and_storage_options, xbasename, xjoin
from .naming import _split_re
from .splits import Split
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import is_local_path, is_relative_path
from .utils.py_utils import glob_pattern_to_regex, string_to_dict
def _get_single_origin_metadata(data_file: str, download_config: Optional[DownloadConfig]=None) -> Tuple[str]:
    data_file, storage_options = _prepare_path_and_storage_options(data_file, download_config=download_config)
    fs, _, _ = get_fs_token_paths(data_file, storage_options=storage_options)
    if isinstance(fs, HfFileSystem):
        resolved_path = fs.resolve_path(data_file)
        return (resolved_path.repo_id, resolved_path.revision)
    elif isinstance(fs, HTTPFileSystem) and data_file.startswith(config.HF_ENDPOINT):
        hffs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=download_config.token)
        data_file = 'hf://' + data_file[len(config.HF_ENDPOINT) + 1:].replace('/resolve/', '@', 1)
        resolved_path = hffs.resolve_path(data_file)
        return (resolved_path.repo_id, resolved_path.revision)
    info = fs.info(data_file)
    for key in ['ETag', 'etag', 'mtime']:
        if key in info:
            return (str(info[key]),)
    return ()