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
def _cloud_path_to_local(self, cloud_path: 'LocalPath') -> Path:
    return self._local_storage_dir / cloud_path._no_prefix