import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
@pytest.mark.parametrize('arr,expected', ((np.array([1, 2], dtype=np.int32), True), (pd.array([1, 2], dtype='Int32'), True), (DummyArray([1, 2], dtype=DummyDtype(numeric=True)), True), (DummyArray([1, 2], dtype=DummyDtype(numeric=False)), False)))
def test_select_dtypes_numeric(self, arr, expected):
    df = DataFrame(arr)
    is_selected = df.select_dtypes(np.number).shape == df.shape
    assert is_selected == expected