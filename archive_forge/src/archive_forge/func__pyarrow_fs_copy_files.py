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
def _pyarrow_fs_copy_files(source, destination, source_filesystem=None, destination_filesystem=None, **kwargs):
    if isinstance(destination_filesystem, pyarrow.fs.S3FileSystem):
        kwargs.setdefault('use_threads', False)
    kwargs.setdefault('chunk_size', 64 * 1024 * 1024)
    return pyarrow.fs.copy_files(source, destination, source_filesystem=source_filesystem, destination_filesystem=destination_filesystem, **kwargs)