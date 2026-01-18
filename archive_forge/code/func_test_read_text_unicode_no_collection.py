from __future__ import annotations
from functools import partial
import pytest
from fsspec.compression import compr
from tlz import concat
from dask import compute, config
from dask.bag.text import read_text
from dask.bytes import utils
from dask.utils import filetexts
def test_read_text_unicode_no_collection(tmp_path):
    data = b'abcd\xc3\xa9'
    fn = tmp_path / 'data.txt'
    with open(fn, 'wb') as f:
        f.write(b'\n'.join([data, data]))
    f = read_text(fn, collection=False)
    result = f[0].compute()
    assert len(result) == 2