import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_pyarrow_dtype_empty_object(self):
    pytest.importorskip('pyarrow')
    expected = pd.DataFrame(columns=[0])
    result = expected.convert_dtypes(dtype_backend='pyarrow')
    tm.assert_frame_equal(result, expected)