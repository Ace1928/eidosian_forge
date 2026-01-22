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
def _glob_checks(self, pattern: str) -> None:
    if '..' in pattern:
        raise CloudPathNotImplementedError("Relative paths with '..' not supported in glob patterns.")
    if pattern.startswith(self.cloud_prefix) or pattern.startswith('/'):
        raise CloudPathNotImplementedError('Non-relative patterns are unsupported')
    if self.drive == '':
        raise CloudPathNotImplementedError(".glob is only supported within a bucket or container; you can use `.iterdir` to list buckets; for example, CloudPath('s3://').iterdir()")