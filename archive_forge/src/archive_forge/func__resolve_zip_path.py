import collections
import contextlib
import itertools
import pathlib
import operator
import re
import warnings
from . import abc
from ._itertools import only
from .compat.py39 import ZipPath
@staticmethod
def _resolve_zip_path(path_str):
    for match in reversed(list(re.finditer('[\\\\/]', path_str))):
        with contextlib.suppress(FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
            inner = path_str[match.end():].replace('\\', '/') + '/'
            yield ZipPath(path_str[:match.start()], inner.lstrip('/'))