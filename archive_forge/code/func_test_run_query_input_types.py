import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
@pytest.mark.parametrize('query', (pa.py_buffer(b'buffer'), b'bytes', 1))
def test_run_query_input_types(tmpdir, query):
    if not isinstance(query, (pa.Buffer, bytes)):
        msg = f"Expected 'pyarrow.Buffer' or bytes, got '{type(query)}'"
        with pytest.raises(TypeError, match=msg):
            substrait.run_query(query)
        return
    msg = 'ParseFromZeroCopyStream failed for substrait.Plan'
    with pytest.raises(OSError, match=msg):
        substrait.run_query(query)