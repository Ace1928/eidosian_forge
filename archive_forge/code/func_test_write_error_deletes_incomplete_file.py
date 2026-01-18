from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
@pytest.mark.pandas
def test_write_error_deletes_incomplete_file(tempdir):
    df = pd.DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6).astype('u1'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.Categorical(list('abc')), 'g': pd.date_range('20130101', periods=3), 'h': pd.date_range('20130101', periods=3, tz='US/Eastern'), 'i': pd.date_range('20130101', periods=3, freq='ns')})
    pdf = pa.Table.from_pandas(df)
    filename = tempdir / 'tmp_file'
    try:
        _write_table(pdf, filename, version='2.4')
    except pa.ArrowException:
        pass
    assert not filename.exists()