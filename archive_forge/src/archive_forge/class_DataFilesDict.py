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
class DataFilesDict(Dict[str, DataFilesList]):
    """
    Dict of split_name -> list of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns :
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover each list is a DataFilesList. It is possible to hash the dictionary
    and get a different hash if and only if at least one file changed.
    For more info, see ``DataFilesList``.

    This is useful for caching Dataset objects that are obtained from a list of data files.

    Changing the order of the keys of this dictionary also doesn't change its hash.
    """

    @classmethod
    def from_local_or_remote(cls, patterns: Dict[str, Union[List[str], DataFilesList]], base_path: Optional[str]=None, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> 'DataFilesDict':
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = DataFilesList.from_local_or_remote(patterns_for_key, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config) if not isinstance(patterns_for_key, DataFilesList) else patterns_for_key
        return out

    @classmethod
    def from_hf_repo(cls, patterns: Dict[str, Union[List[str], DataFilesList]], dataset_info: huggingface_hub.hf_api.DatasetInfo, base_path: Optional[str]=None, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> 'DataFilesDict':
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = DataFilesList.from_hf_repo(patterns_for_key, dataset_info=dataset_info, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config) if not isinstance(patterns_for_key, DataFilesList) else patterns_for_key
        return out

    @classmethod
    def from_patterns(cls, patterns: Dict[str, Union[List[str], DataFilesList]], base_path: Optional[str]=None, allowed_extensions: Optional[List[str]]=None, download_config: Optional[DownloadConfig]=None) -> 'DataFilesDict':
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = DataFilesList.from_patterns(patterns_for_key, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config) if not isinstance(patterns_for_key, DataFilesList) else patterns_for_key
        return out

    def filter_extensions(self, extensions: List[str]) -> 'DataFilesDict':
        out = type(self)()
        for key, data_files_list in self.items():
            out[key] = data_files_list.filter_extensions(extensions)
        return out