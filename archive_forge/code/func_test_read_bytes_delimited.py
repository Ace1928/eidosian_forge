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
def test_read_bytes_delimited():
    with filetexts(files, mode='b'):
        for bs in [5, 15, 45, '1.5 kB']:
            _, values = read_bytes('.test.accounts*', blocksize=bs, delimiter=b'\n')
            _, values2 = read_bytes('.test.accounts*', blocksize=bs, delimiter=b'foo')
            assert [a.key for a in concat(values)] != [b.key for b in concat(values2)]
            results = compute(*concat(values))
            res = [r for r in results if r]
            assert all((r.endswith(b'\n') for r in res))
            ourlines = b''.join(res).split(b'\n')
            testlines = b''.join((files[k] for k in sorted(files))).split(b'\n')
            assert ourlines == testlines
            d = b'}'
            _, values = read_bytes('.test.accounts*', blocksize=bs, delimiter=d)
            results = compute(*concat(values))
            res = [r for r in results if r]
            assert sum((r.endswith(b'}') for r in res)) == len(res) - 2
            ours = b''.join(res)
            test = b''.join((files[v] for v in sorted(files)))
            assert ours == test