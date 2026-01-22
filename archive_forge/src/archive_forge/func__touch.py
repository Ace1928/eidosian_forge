import atexit
from hashlib import md5
import mimetypes
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from ..client import Client
from ..enums import FileCacheMode
from .localpath import LocalPath
def _touch(self, cloud_path: 'LocalPath', exist_ok: bool=True) -> None:
    local_storage_path = self._cloud_path_to_local(cloud_path)
    if local_storage_path.exists() and (not exist_ok):
        raise FileExistsError(f'File exists: {cloud_path}')
    local_storage_path.parent.mkdir(exist_ok=True, parents=True)
    local_storage_path.touch()