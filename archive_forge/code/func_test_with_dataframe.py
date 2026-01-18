import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_with_dataframe():
    arr = SparseArray([0, 1], fill_value=0)
    df = pd.DataFrame([[1, 2], [3, 4]])
    result = arr.__add__(df)
    assert result is NotImplemented