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
def _write_multiple_hdfs_pq_files(self, tmpdir):
    import pyarrow.parquet as pq
    nfiles = 10
    size = 5
    test_data = []
    for i in range(nfiles):
        df = _test_dataframe(size, seed=i)
        df['index'] = np.arange(i * size, (i + 1) * size)
        df['uint32'] = df['uint32'].astype(np.int64)
        path = pjoin(tmpdir, '{}.parquet'.format(i))
        table = pa.Table.from_pandas(df, preserve_index=False)
        with self.hdfs.open(path, 'wb') as f:
            pq.write_table(table, f)
        test_data.append(table)
    expected = pa.concat_tables(test_data)
    return expected