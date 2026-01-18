from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_preserve_categorical_dtype():
    df = DataFrame({'A': [1, 2, 1, 1, 2], 'B': [10, 16, 22, 28, 34], 'C1': Categorical(list('abaab'), categories=list('bac'), ordered=False), 'C2': Categorical(list('abaab'), categories=list('bac'), ordered=True)})
    exp_full = DataFrame({'A': [2.0, 1.0, np.nan], 'B': [25.0, 20.0, np.nan], 'C1': Categorical(list('bac'), categories=list('bac'), ordered=False), 'C2': Categorical(list('bac'), categories=list('bac'), ordered=True)})
    for col in ['C1', 'C2']:
        result1 = df.groupby(by=col, as_index=False, observed=False).mean(numeric_only=True)
        result2 = df.groupby(by=col, as_index=True, observed=False).mean(numeric_only=True).reset_index()
        expected = exp_full.reindex(columns=result1.columns)
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)