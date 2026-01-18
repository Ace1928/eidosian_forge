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
def test_s3fs_wrong_region():
    from pyarrow.fs import S3FileSystem
    fs = S3FileSystem(region='eu-north-1', anonymous=True)
    msg = "When getting information for bucket 'voltrondata-labs-datasets': AWS Error UNKNOWN \\(HTTP status 301\\) during HeadBucket operation: No response body. Looks like the configured region is 'eu-north-1' while the bucket is located in 'us-east-2'.|NETWORK_CONNECTION"
    with pytest.raises(OSError, match=msg) as exc:
        fs.get_file_info('voltrondata-labs-datasets')
    if 'NETWORK_CONNECTION' in str(exc.value):
        return
    fs = S3FileSystem(region='us-east-2', anonymous=True)
    fs.get_file_info('voltrondata-labs-datasets')