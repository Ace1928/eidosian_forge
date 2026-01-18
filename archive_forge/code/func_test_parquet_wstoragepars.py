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
def test_parquet_wstoragepars(s3, s3so, engine):
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')
    np = pytest.importorskip('numpy')
    url = 's3://%s/test.parquet' % test_bucket_name
    data = pd.DataFrame({'i32': np.array([0, 5, 2, 5])})
    df = dd.from_pandas(data, chunksize=500)
    df.to_parquet(url, engine=engine, write_index=False, storage_options=s3so, write_metadata_file=True)
    dd.read_parquet(url, engine=engine, storage_options={'default_fill_cache': False, **s3so})
    assert s3.current().default_fill_cache is False
    dd.read_parquet(url, engine=engine, storage_options={'default_fill_cache': True, **s3so})
    assert s3.current().default_fill_cache is True
    dd.read_parquet(url, engine=engine, storage_options={'default_block_size': 2 ** 20, **s3so})
    assert s3.current().default_block_size == 2 ** 20
    with s3.current().open(url + '/_metadata') as f:
        assert f.blocksize == 2 ** 20