import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def test_orc_writer_with_null_arrays(tempdir):
    from pyarrow import orc
    path = str(tempdir / 'test.orc')
    a = pa.array([1, None, 3, None])
    b = pa.array([None, None, None, None])
    table = pa.table({'int64': a, 'utf8': b})
    with pytest.raises(pa.ArrowNotImplementedError):
        orc.write_table(table, path)