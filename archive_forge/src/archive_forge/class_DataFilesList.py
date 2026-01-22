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
class DataFilesList(List[str]):
    """
    List of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns :
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover DataFilesList has an additional attribute ``origin_metadata``.
    It can store:
    - the last modified time of local files
    - ETag of remote files
    - commit sha of a dataset repository

    Thanks to this additional attribute, it is possible to hash the list
    and get a different hash if and only if at least one file changed.
    This is useful for caching Dataset objects that are obtained from a list of data files.
    """

    def __init__(self, data_files: List[str], origin_metadata: List[Tuple[str]]):
        super().__init__(data_files)
        self.origin_metadata = origin_metadata

    def __add__(self, other):
        return DataFilesList([*self, *other], self.origin_metadata + other.origin_metadata)

    @classmethod
    def from_hf_repo(cls, patterns: List[str], dataset_info: huggingface_hub.hf_api.DatasetInfo, base_path: Optional[str]=None, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> 'DataFilesList':
        base_path = f'hf://datasets/{dataset_info.id}@{dataset_info.sha}/{base_path or ''}'.rstrip('/')
        return cls.from_patterns(patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config)

    @classmethod
    def from_local_or_remote(cls, patterns: List[str], base_path: Optional[str]=None, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> 'DataFilesList':
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        return cls.from_patterns(patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config)

    @classmethod
    def from_patterns(cls, patterns: List[str], base_path: Optional[str]=None, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> 'DataFilesList':
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern in patterns:
            try:
                data_files.extend(resolve_pattern(pattern, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config))
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return cls(data_files, origin_metadata)

    def filter_extensions(self, extensions: List[str]) -> 'DataFilesList':
        pattern = '|'.join(('\\' + ext for ext in extensions))
        pattern = re.compile(f'.*({pattern})(\\..+)?$')
        return DataFilesList([data_file for data_file in self if pattern.match(data_file)], origin_metadata=self.origin_metadata)