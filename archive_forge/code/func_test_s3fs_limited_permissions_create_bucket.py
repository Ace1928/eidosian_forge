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
def test_s3fs_limited_permissions_create_bucket(s3_server):
    from pyarrow.fs import S3FileSystem
    _configure_s3_limited_user(s3_server, _minio_limited_policy)
    host, port, _, _ = s3_server['connection']
    fs = S3FileSystem(access_key='limited', secret_key='limited123', endpoint_override='{}:{}'.format(host, port), scheme='http')
    fs.create_dir('existing-bucket/test')
    with pytest.raises(pa.ArrowIOError, match="Bucket 'new-bucket' not found"):
        fs.create_dir('new-bucket')
    with pytest.raises(pa.ArrowIOError, match='Would delete bucket'):
        fs.delete_dir('existing-bucket')