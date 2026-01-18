import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
def test_pyarrow_array_to_numpy_and_mask(np_dtype_to_arrays):
    """
    Test conversion from pyarrow array to numpy array.

    Modifies the pyarrow buffer to contain padding and offset, which are
    considered valid buffers by pyarrow.

    Also tests empty pyarrow arrays with non empty buffers.
    See https://github.com/pandas-dev/pandas/issues/40896
    """
    np_dtype, pa_array, np_expected, mask_expected = np_dtype_to_arrays
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)
    mask_buffer = pa_array.buffers()[0]
    data_buffer = pa_array.buffers()[1]
    data_buffer_bytes = pa_array.buffers()[1].to_pybytes()
    data_buffer_trail = pa.py_buffer(data_buffer_bytes + b'\x00')
    pa_array_trail = pa.Array.from_buffers(type=pa_array.type, length=len(pa_array), buffers=[mask_buffer, data_buffer_trail], offset=pa_array.offset)
    pa_array_trail.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_trail, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)
    offset = b'\x00' * (pa_array.type.bit_width // 8)
    data_buffer_offset = pa.py_buffer(offset + data_buffer_bytes)
    mask_buffer_offset = pa.py_buffer(b'\x0e')
    pa_array_offset = pa.Array.from_buffers(type=pa_array.type, length=len(pa_array), buffers=[mask_buffer_offset, data_buffer_offset], offset=pa_array.offset + 1)
    pa_array_offset.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)
    np_expected_empty = np.array([], dtype=np_dtype)
    mask_expected_empty = np.array([], dtype=np.bool_)
    pa_array_offset = pa.Array.from_buffers(type=pa_array.type, length=0, buffers=[mask_buffer, data_buffer], offset=pa_array.offset)
    pa_array_offset.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected_empty)
    tm.assert_numpy_array_equal(mask, mask_expected_empty)