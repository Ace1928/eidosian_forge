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
def _dispatch_to_local_cache_path(self, func: str, *args, **kwargs) -> Any:
    self._refresh_cache()
    path_version = self._local.__getattribute__(func)
    if callable(path_version):
        path_version = path_version(*args, **kwargs)
    if isinstance(path_version, (PosixPath, WindowsPath)):
        path_version = path_version.resolve()
        return self._new_cloudpath(path_version)
    else:
        return path_version