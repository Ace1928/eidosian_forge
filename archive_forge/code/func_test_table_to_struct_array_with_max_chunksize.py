from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_to_struct_array_with_max_chunksize():
    table = pa.Table.from_arrays([pa.array([1, None], type=pa.int32()), pa.array([None, 1.0], type=pa.float32())], ['ints', 'floats'])
    result = table.to_struct_array(max_chunksize=1)
    assert result.equals(pa.chunked_array([[{'ints': 1}], [{'floats': 1.0}]], type=pa.struct([('ints', pa.int32()), ('floats', pa.float32())])))