import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_categorical1(self, frame_or_series):
    obj = frame_or_series(['a', 'b', 'c', 'd'], dtype='category')
    rt = obj.shift(1).shift(-1)
    tm.assert_equal(obj.iloc[:-1], rt.dropna())

    def get_cat_values(ndframe):
        return ndframe._mgr.arrays[0]
    cat = get_cat_values(obj)
    sp1 = obj.shift(1)
    tm.assert_index_equal(obj.index, sp1.index)
    assert np.all(get_cat_values(sp1).codes[:1] == -1)
    assert np.all(cat.codes[:-1] == get_cat_values(sp1).codes[1:])
    sn2 = obj.shift(-2)
    tm.assert_index_equal(obj.index, sn2.index)
    assert np.all(get_cat_values(sn2).codes[-2:] == -1)
    assert np.all(cat.codes[2:] == get_cat_values(sn2).codes[:-2])
    tm.assert_index_equal(cat.categories, get_cat_values(sp1).categories)
    tm.assert_index_equal(cat.categories, get_cat_values(sn2).categories)