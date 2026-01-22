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
def _local_to_cloud_path(self, local_path: Union[str, os.PathLike]) -> 'LocalPath':
    local_path = Path(local_path)
    cloud_prefix = self._cloud_meta.path_class.cloud_prefix
    return self.CloudPath(f'{cloud_prefix}{PurePosixPath(local_path.relative_to(self._local_storage_dir))}')