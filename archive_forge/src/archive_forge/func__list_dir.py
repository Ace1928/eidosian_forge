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
def _list_dir(self, cloud_path: 'LocalPath', recursive=False) -> Iterable[Tuple['LocalPath', bool]]:
    if recursive:
        return ((self._local_to_cloud_path(obj), obj.is_dir()) for obj in self._cloud_path_to_local(cloud_path).glob('**/*'))
    return ((self._local_to_cloud_path(obj), obj.is_dir()) for obj in self._cloud_path_to_local(cloud_path).iterdir())