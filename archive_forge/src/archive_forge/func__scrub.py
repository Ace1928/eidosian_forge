from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from fsspec.core import expand_paths_if_needed, get_fs_token_paths, stringify_path
from fsspec.spec import AbstractFileSystem
from dask import config
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key, parse_bytes
def _scrub(i, p):
    p = p.replace(fs.sep, '/')
    if p == '':
        return '.'
    if p[-1] == '/':
        p = p[:-1]
    if i > 0 and p[0] == '/':
        p = p[1:]
    return p