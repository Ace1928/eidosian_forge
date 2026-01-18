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
@pytest.mark.parametrize('mode', ['rt', 'rb'])
@pytest.mark.parametrize('fmt', list(compr))
def test_open_files_compression(mode, fmt):
    if fmt not in compress:
        pytest.skip('compression function not provided')
    files2 = valmap(compress[fmt], files)
    with filetexts(files2, mode='b'):
        myfiles = open_files('.test.accounts.*', mode=mode, compression=fmt)
        data = []
        for file in myfiles:
            with file as f:
                data.append(f.read())
        sol = [files[k] for k in sorted(files)]
        if mode == 'rt':
            sol = [b.decode() for b in sol]
        assert list(data) == sol