import dataclasses
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from ray._private.storage import _get_storage_uri
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.syncer import SyncConfig, Syncer, _BackgroundSyncer
from ray.train.constants import _get_defaults_results_dir
def _list_at_fs_path(fs: pyarrow.fs.FileSystem, fs_path: str, file_filter: Callable[[pyarrow.fs.FileInfo], bool]=lambda x: True) -> List[str]:
    """Returns the list of filenames at (fs, fs_path), similar to os.listdir.

    If the path doesn't exist, returns an empty list.
    """
    selector = pyarrow.fs.FileSelector(fs_path, allow_not_found=True, recursive=False)
    return [os.path.relpath(file_info.path.lstrip('/'), start=fs_path.lstrip('/')) for file_info in fs.get_file_info(selector) if file_filter(file_info)]