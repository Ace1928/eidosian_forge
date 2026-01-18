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
def test_slice_iloc():
    df = makeDataFrame()
    column = IntSlider(start=10, end=40)
    ds = Dataset(df)
    transform = util.transform.df_dim('*').iloc[:column].mean(axis=0)
    params = list(transform.params.values())
    assert len(params) == 1
    assert params[0] == column.param.value
    df1 = transform.apply(ds, keep_index=True, compute=False)
    df2 = df.iloc[:10].mean(axis=0)
    pd.testing.assert_series_equal(df1, df2)