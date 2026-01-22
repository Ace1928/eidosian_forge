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
def _refresh_cache(self, force_overwrite_from_cloud: bool=False) -> None:
    try:
        stats = self.stat()
    except NoStatError:
        return
    if not self._local.exists() or self._local.stat().st_mtime < stats.st_mtime or force_overwrite_from_cloud:
        self._local.parent.mkdir(parents=True, exist_ok=True)
        self.download_to(self._local)
        os.utime(self._local, times=(stats.st_mtime, stats.st_mtime))
    if self._dirty:
        raise OverwriteDirtyFileError(f'Local file ({self._local}) for cloud path ({self}) has been changed by your code, but is being requested for download from cloud. Either (1) push your changes to the cloud, (2) remove the local file, or (3) pass `force_overwrite_from_cloud=True` to overwrite.')
    if self._local.stat().st_mtime > stats.st_mtime:
        raise OverwriteNewerLocalError(f'Local file ({self._local}) for cloud path ({self}) is newer on disk, but is being requested for download from cloud. Either (1) push your changes to the cloud, (2) remove the local file, or (3) pass `force_overwrite_from_cloud=True` to overwrite.')