import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
def _upload_local_to_cloud(self, force_overwrite_to_cloud: bool=False) -> Self:
    """Uploads cache file at self._local to the cloud"""
    if self._local.is_dir():
        raise ValueError('Only individual files can be uploaded to the cloud')
    uploaded = self._upload_file_to_cloud(self._local, force_overwrite_to_cloud=force_overwrite_to_cloud)
    stats = self.stat()
    os.utime(self._local, times=(stats.st_mtime, stats.st_mtime))
    self._dirty = False
    self._handle = None
    return uploaded