import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
@pytest.mark.xfail(reason='legacy.FileSystem not supported with ParquetDataset due to legacy path being removed in PyArrow 15.0.0.', raises=TypeError)
@pytest.mark.pandas
@pytest.mark.parquet
def test_read_write_parquet_files_with_uri(self):
    import pyarrow.parquet as pq
    tmpdir = pjoin(self.tmp_path, 'uri-parquet-' + guid())
    self.hdfs.mkdir(tmpdir)
    path = _get_hdfs_uri(pjoin(tmpdir, 'test.parquet'))
    size = 5
    df = _test_dataframe(size, seed=0)
    df['uint32'] = df['uint32'].astype(np.int64)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, filesystem=self.hdfs)
    result = pq.read_table(path, filesystem=self.hdfs).to_pandas()
    assert_frame_equal(result, df)