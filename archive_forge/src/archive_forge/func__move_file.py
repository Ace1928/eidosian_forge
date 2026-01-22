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
def _move_file(self, src: 'LocalPath', dst: 'LocalPath', remove_src: bool=True) -> 'LocalPath':
    self._cloud_path_to_local(dst).parent.mkdir(exist_ok=True, parents=True)
    if remove_src:
        self._cloud_path_to_local(src).replace(self._cloud_path_to_local(dst))
    else:
        shutil.copy(self._cloud_path_to_local(src), self._cloud_path_to_local(dst))
    return dst