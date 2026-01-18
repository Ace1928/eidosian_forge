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
@pytest.mark.skipif(sys.platform == 'win32', reason='pathlib and moto clash on windows')
def test_with_paths():
    with filetexts(files, mode='b'):
        url = pathlib.Path('./.test.accounts.*')
        sample, values = read_bytes(url, blocksize=None)
        assert sum(map(len, values)) == len(files)
    with pytest.raises(OSError):
        url = pathlib.Path('file://.test.accounts.*')
        read_bytes(url, blocksize=None)