import contextlib
import glob
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union
import pyarrow.fs
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.storage import _download_from_fs_path, _exists_at_fs_path
from ray.util.annotations import PublicAPI
def _list_existing_del_locks(path: str) -> List[str]:
    """List all the deletion lock files for a file/directory at `path`.

    For example, if 2 checkpoints are being read via `as_directory`,
    then this should return a list of 2 deletion lock files.
    """
    return list(glob.glob(f'{_get_del_lock_path(path, suffix='*')}'))