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
def to_directory(self, path: Optional[Union[str, os.PathLike]]=None) -> str:
    """Write checkpoint data to a local directory.

        *If multiple processes on the same node call this method simultaneously,*
        only a single process will perform the download, while the others
        wait for the download to finish. Once the download finishes, all processes
        receive the same local directory to read from.

        Args:
            path: Target directory to download data to. If not specified,
                this method will use a temporary directory.

        Returns:
            str: Directory containing checkpoint data.
        """
    user_provided_path = path is not None
    local_path = path if user_provided_path else self._get_temporary_checkpoint_dir()
    local_path = os.path.normpath(os.path.expanduser(str(local_path)))
    os.makedirs(local_path, exist_ok=True)
    try:
        with TempFileLock(local_path, timeout=0):
            _download_from_fs_path(fs=self.filesystem, fs_path=self.path, local_path=local_path)
    except TimeoutError:
        with TempFileLock(local_path, timeout=-1):
            pass
        if not os.path.exists(local_path):
            raise RuntimeError(f'Checkpoint directory {local_path} does not exist, even though it should have been created by another process. Please raise an issue on GitHub: https://github.com/ray-project/ray/issues')
    return local_path