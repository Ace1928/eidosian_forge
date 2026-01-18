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
def test_parquet_append(s3, engine, s3so):
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')
    np = pytest.importorskip('numpy')
    if dd._dask_expr_enabled():
        pytest.skip('need convert string option')
    url = 's3://%s/test.parquet.append' % test_bucket_name
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32), 'i64': np.arange(1000, dtype=np.int64), 'f': np.arange(1000, dtype=np.float64), 'bhello': np.random.choice(['hello', 'you', 'people'], size=1000).astype('O')})
    df = dd.from_pandas(data, chunksize=500)
    df.to_parquet(url, engine=engine, storage_options=s3so, write_index=False, write_metadata_file=True)
    df.to_parquet(url, engine=engine, storage_options=s3so, write_index=False, append=True, ignore_divisions=True)
    files = [f.split('/')[-1] for f in s3.ls(url)]
    assert '_common_metadata' in files
    assert '_metadata' in files
    assert 'part.0.parquet' in files
    df2 = dd.read_parquet(url, index=False, engine=engine, storage_options=s3so)
    dd.utils.assert_eq(pd.concat([data, data]), df2, check_index=False)