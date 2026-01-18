from __future__ import annotations
import gzip
import os
import pathlib
import sys
from functools import partial
from time import sleep
import cloudpickle
import pytest
from fsspec.compression import compr
from fsspec.core import open_files
from fsspec.implementations.local import LocalFileSystem
from tlz import concat, valmap
from dask import compute
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.utils import filetexts
def test_read_bytes_block():
    with filetexts(files, mode='b'):
        for bs in [5, 15, 45, 1500]:
            sample, vals = read_bytes('.test.account*', blocksize=bs)
            assert list(map(len, vals)) == [max(len(v) // bs, 1) for v in files.values()]
            results = compute(*concat(vals))
            assert sum((len(r) for r in results)) == sum((len(v) for v in files.values()))
            ourlines = b''.join(results).split(b'\n')
            testlines = b''.join(files.values()).split(b'\n')
            assert set(ourlines) == set(testlines)