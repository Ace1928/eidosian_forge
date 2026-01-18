import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_pyarrow_backend_no_conversion(self):
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': [1, 2], 'b': 1.5, 'c': True, 'd': 'x'})
    expected = df.copy()
    result = df.convert_dtypes(convert_floating=False, convert_integer=False, convert_boolean=False, convert_string=False, dtype_backend='pyarrow')
    tm.assert_frame_equal(result, expected)