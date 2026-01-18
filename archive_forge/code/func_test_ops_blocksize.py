from __future__ import annotations
import os
import subprocess
import sys
import time
import fsspec
import pytest
from fsspec.core import open_files
from packaging.version import parse as parse_version
import dask.bag as db
from dask.utils import tmpdir
def test_ops_blocksize(dir_server):
    root = 'http://localhost:8999/'
    fn = files[0]
    f = open_files(root + fn, block_size=2)[0]
    with open(os.path.join(dir_server, fn), 'rb') as expected:
        expected = expected.read()
        with f as f:
            assert f.read() == expected
            assert f.size == len(expected)
        fn = files[1]
        f = open_files(root + fn, block_size=2)[0]
        with f as f:
            if parse_version(fsspec.__version__) < parse_version('2021.11.1'):
                with pytest.raises(ValueError):
                    assert f.read(10) == expected[:10]
            else:
                assert f.read(10) == expected[:10]