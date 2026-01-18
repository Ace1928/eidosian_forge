import numpy as np
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider, RadioButtonGroup, TextInput
from holoviews import Dataset, util
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import Curve, Image
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import ParamMethod, Params
def test_slice_loc():
    df = makeDataFrame()
    df.index = np.arange(5, len(df) + 5)
    column = IntSlider(start=10, end=40)
    ds = Dataset(df)
    transform = util.transform.df_dim('*').loc[:column].mean(axis=0)
    params = list(transform.params.values())
    assert len(params) == 1
    assert params[0] == column.param.value
    df1 = transform.apply(ds, keep_index=True, compute=False)
    df2 = df.loc[5:10].mean(axis=0)
    pd.testing.assert_series_equal(df1, df2)
    df3 = df.iloc[5:10].mean(axis=0)
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(df1, df3)