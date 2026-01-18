from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('normalize, expected_label', [(False, 'count'), (True, 'proportion')])
def test_result_label_duplicates(normalize, expected_label):
    gb = DataFrame([[1, 2, 3]], columns=['a', 'b', expected_label]).groupby('a', as_index=False)
    msg = f"Column label '{expected_label}' is duplicate of result column"
    with pytest.raises(ValueError, match=msg):
        gb.value_counts(normalize=normalize)