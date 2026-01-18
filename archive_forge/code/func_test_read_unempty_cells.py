import functools
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
def test_read_unempty_cells():
    expected = pd.DataFrame([1, np.nan, 3, np.nan, 5], columns=['Column 1'])
    result = pd.read_excel('test_unempty_cells.ods')
    tm.assert_frame_equal(result, expected)