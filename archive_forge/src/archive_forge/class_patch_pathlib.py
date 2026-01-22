import collections
import io
import locale
import logging
import os
import os.path as P
import pathlib
import urllib.parse
import warnings
import smart_open.local_file as so_file
import smart_open.compression as so_compression
from smart_open import doctools
from smart_open import transport
from smart_open.compression import register_compressor  # noqa: F401
from smart_open.utils import check_kwargs as _check_kwargs  # noqa: F401
from smart_open.utils import inspect_kwargs as _inspect_kwargs  # noqa: F401
class patch_pathlib(object):
    """Replace `Path.open` with `smart_open.open`"""

    def __init__(self):
        self.old_impl = _patch_pathlib(open)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _patch_pathlib(self.old_impl)