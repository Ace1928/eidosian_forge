import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclass_unstack(self):
    df = tm.SubclassedDataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'b', 'c'], columns=['X', 'Y', 'Z'])
    res = df.unstack()
    exp = tm.SubclassedSeries([1, 4, 7, 2, 5, 8, 3, 6, 9], index=[list('XXXYYYZZZ'), list('abcabcabc')])
    tm.assert_series_equal(res, exp)