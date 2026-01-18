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
def test_s3_proxy_options(monkeypatch, pickle_module):
    from pyarrow.fs import S3FileSystem
    proxy_opts_1_dict = {'scheme': 'http', 'host': 'localhost', 'port': 8999}
    proxy_opts_1_str = 'http://localhost:8999'
    proxy_opts_2_dict = {'scheme': 'https', 'host': 'localhost', 'port': 8080}
    proxy_opts_2_str = 'https://localhost:8080'
    fs = S3FileSystem(proxy_options=proxy_opts_1_dict)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs = S3FileSystem(proxy_options=proxy_opts_1_str)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert isinstance(fs, S3FileSystem)
    assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    assert fs1 == fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) == fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert fs1 == fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) == fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_1_str)
    assert fs1 == fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) == fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_2_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 == fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) == fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_1_str)
    assert fs1 == fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) == fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 == fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) == fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) == fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    fs1 = S3FileSystem(proxy_options=proxy_opts_2_str)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle_module.loads(pickle_module.dumps(fs1)) != fs2
    assert pickle_module.loads(pickle_module.dumps(fs2)) != fs1
    with pytest.raises(TypeError):
        S3FileSystem(proxy_options=('http', 'localhost', 9090))
    with pytest.raises(KeyError):
        S3FileSystem(proxy_options={'host': 'localhost', 'port': 9090})
    with pytest.raises(KeyError):
        S3FileSystem(proxy_options={'scheme': 'https', 'port': 9090})
    with pytest.raises(KeyError):
        S3FileSystem(proxy_options={'scheme': 'http', 'host': 'localhost'})
    with pytest.raises(pa.ArrowInvalid):
        S3FileSystem(proxy_options='httpsB://localhost:9000')
    with pytest.raises(pa.ArrowInvalid):
        S3FileSystem(proxy_options={'scheme': 'httpA', 'host': 'localhost', 'port': 8999})