import functools
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
def test_read_writer_table():
    index = pd.Index(['Row 1', 'Row 2', 'Row 3'], name='Header')
    expected = pd.DataFrame([[1, np.nan, 7], [2, np.nan, 8], [3, np.nan, 9]], index=index, columns=['Column 1', 'Unnamed: 2', 'Column 3'])
    result = pd.read_excel('writertable.odt', sheet_name='Table1', index_col=0)
    tm.assert_frame_equal(result, expected)