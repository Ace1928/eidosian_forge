from __future__ import annotations
import json
from typing import Protocol, runtime_checkable
from uuid import uuid4
import fsspec
import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from packaging.version import parse as parse_version
def _is_local_fs_pyarrow(fs):
    """Check if a pyarrow-based file-system is local"""
    if fs:
        if hasattr(fs, 'fs'):
            return _is_local_fs_pyarrow(fs.fs)
        elif hasattr(fs, 'type_name'):
            return fs.type_name == 'local'
    return False