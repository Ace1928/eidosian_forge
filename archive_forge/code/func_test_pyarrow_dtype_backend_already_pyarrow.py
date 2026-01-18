import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_pyarrow_dtype_backend_already_pyarrow(self):
    pytest.importorskip('pyarrow')
    expected = pd.DataFrame([1, 2, 3], dtype='int64[pyarrow]')
    result = expected.convert_dtypes(dtype_backend='pyarrow')
    tm.assert_frame_equal(result, expected)