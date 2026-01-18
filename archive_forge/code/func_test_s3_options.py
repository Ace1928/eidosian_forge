from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
@pytest.mark.s3
def test_s3_options(pickle_module):
    from pyarrow.fs import AwsDefaultS3RetryStrategy, AwsStandardS3RetryStrategy, S3FileSystem, S3RetryStrategy
    fs = S3FileSystem(access_key='access', secret_key='secret', session_token='token', region='us-east-2', scheme='https', endpoint_override='localhost:8999')
    assert isinstance(fs, S3FileSystem)
    assert fs.region == 'us-east-2'
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs = S3FileSystem(role_arn='role', session_name='session', external_id='id', load_frequency=100)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs = S3FileSystem(retry_strategy=AwsStandardS3RetryStrategy(max_attempts=5))
    assert isinstance(fs, S3FileSystem)
    fs = S3FileSystem(retry_strategy=AwsDefaultS3RetryStrategy(max_attempts=5))
    assert isinstance(fs, S3FileSystem)
    fs2 = S3FileSystem(role_arn='role')
    assert isinstance(fs2, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs2
    assert fs2 != fs
    fs = S3FileSystem(anonymous=True)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs = S3FileSystem(background_writes=True)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs2 = S3FileSystem(background_writes=True, default_metadata={'ACL': 'authenticated-read', 'Content-Type': 'text/plain'})
    assert isinstance(fs2, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs2
    assert fs2 != fs
    fs = S3FileSystem(allow_bucket_creation=True, allow_bucket_deletion=True)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs = S3FileSystem(request_timeout=0.5, connect_timeout=0.25)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs2 = S3FileSystem(request_timeout=0.25, connect_timeout=0.5)
    assert isinstance(fs2, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs2
    assert fs2 != fs
    with pytest.raises(ValueError):
        S3FileSystem(access_key='access')
    with pytest.raises(ValueError):
        S3FileSystem(secret_key='secret')
    with pytest.raises(ValueError):
        S3FileSystem(access_key='access', session_token='token')
    with pytest.raises(ValueError):
        S3FileSystem(secret_key='secret', session_token='token')
    with pytest.raises(ValueError):
        S3FileSystem(access_key='access', secret_key='secret', role_arn='arn')
    with pytest.raises(ValueError):
        S3FileSystem(access_key='access', secret_key='secret', anonymous=True)
    with pytest.raises(ValueError):
        S3FileSystem(role_arn='arn', anonymous=True)
    with pytest.raises(ValueError):
        S3FileSystem(default_metadata=['foo', 'bar'])
    with pytest.raises(ValueError):
        S3FileSystem(retry_strategy=S3RetryStrategy())