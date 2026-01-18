import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_pivot_df_multiindex_index_none():
    df = pd.DataFrame([['A', 'A1', 'label1', 1], ['A', 'A2', 'label2', 2], ['B', 'A1', 'label1', 3], ['B', 'A2', 'label2', 4]], columns=['index_1', 'index_2', 'label', 'value'])
    df = df.set_index(['index_1', 'index_2'])
    result = df.pivot(columns='label', values='value')
    expected = pd.DataFrame([[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan], [np.nan, 4.0]], index=df.index, columns=Index(['label1', 'label2'], name='label'))
    tm.assert_frame_equal(result, expected)