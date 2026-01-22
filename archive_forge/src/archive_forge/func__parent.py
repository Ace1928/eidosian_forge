import datetime
import io
import logging
import os
import os.path as osp
import re
import shutil
import stat
import tempfile
from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.utils import isfilelike, stringify_path
@classmethod
def _parent(cls, path):
    path = cls._strip_protocol(path).rstrip('/')
    if '/' in path:
        return path.rsplit('/', 1)[0]
    else:
        return cls.root_marker