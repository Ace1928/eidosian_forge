import weakref
import numpy as np
import pyarrow as pa
from pyarrow.lib import StringBuilder
def test_string_builder_append():
    sbuilder = StringBuilder()
    sbuilder.append(b'a byte string')
    sbuilder.append('a string')
    sbuilder.append(np.nan)
    sbuilder.append(None)
    assert len(sbuilder) == 4
    assert sbuilder.null_count == 2
    arr = sbuilder.finish()
    assert len(sbuilder) == 0
    assert isinstance(arr, pa.Array)
    assert arr.null_count == 2
    assert arr.type == 'str'
    expected = ['a byte string', 'a string', None, None]
    assert arr.to_pylist() == expected