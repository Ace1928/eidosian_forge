from __future__ import annotations
import io
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
import pytest
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths, open_files
from s3fs import S3FileSystem as DaskS3FileSystem
from tlz import concat, valmap
from dask import compute
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
def test_read_bytes_non_existing_glob(s3, s3so):
    with pytest.raises(IOError):
        read_bytes('s3://' + test_bucket_name + '/non-existing/*', **s3so)