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
def test_s3_real_aws_region_selection():
    fs, path = FileSystem.from_uri('s3://mf-nwp-models/README.txt')
    assert fs.region == 'eu-west-1'
    with fs.open_input_stream(path) as f:
        assert b'Meteo-France Atmospheric models on AWS' in f.read(50)
    fs, path = FileSystem.from_uri('s3://mf-nwp-models/README.txt?region=us-east-2')
    assert fs.region == 'us-east-2'
    with pytest.raises(IOError, match="Bucket '.*' not found"):
        FileSystem.from_uri('s3://x-arrow-nonexistent-bucket')
    fs, path = FileSystem.from_uri('s3://x-arrow-nonexistent-bucket?region=us-east-3')
    assert fs.region == 'us-east-3'