import functools
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
def test_read_newlines_between_xml_elements_table():
    expected = pd.DataFrame([[1.0, 4.0, 7], [np.nan, np.nan, 8], [3.0, 6.0, 9]], columns=['Column 1', 'Column 2', 'Column 3'])
    result = pd.read_excel('test_newlines.ods')
    tm.assert_frame_equal(result, expected)