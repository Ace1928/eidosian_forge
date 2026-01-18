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
def test_s3_finalize_region_resolver():
    code = 'if 1:\n        import pytest\n        from pyarrow.fs import resolve_s3_region, ensure_s3_initialized, finalize_s3\n\n        resolve_s3_region(\'mf-nwp-models\')\n\n        finalize_s3()\n\n        # Testing both cached and uncached accesses\n        with pytest.raises(ValueError, match="S3 .* finalized"):\n            resolve_s3_region(\'mf-nwp-models\')\n        with pytest.raises(ValueError, match="S3 .* finalized"):\n            resolve_s3_region(\'voltrondata-labs-datasets\')\n        '
    subprocess.check_call([sys.executable, '-c', code])