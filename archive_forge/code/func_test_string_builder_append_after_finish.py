import weakref
import numpy as np
import pyarrow as pa
from pyarrow.lib import StringBuilder
def test_string_builder_append_after_finish():
    sbuilder = StringBuilder()
    sbuilder.append_values([np.nan, None, 'text', None, 'other text'])
    arr = sbuilder.finish()
    sbuilder.append('No effect')
    expected = [None, None, 'text', None, 'other text']
    assert arr.to_pylist() == expected