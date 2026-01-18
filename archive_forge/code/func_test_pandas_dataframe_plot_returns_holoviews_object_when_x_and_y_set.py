from unittest import TestCase, SkipTest
import holoviews as hv
import pandas as pd
import pytest
from packaging.version import Version
from parameterized import parameterized
from hvplot.converter import HoloViewsConverter
from hvplot.plotting import plot
from hvplot.tests.util import makeDataFrame
@parameterized.expand(x_y_mapping)
def test_pandas_dataframe_plot_returns_holoviews_object_when_x_and_y_set(self, kind, el):
    df = pd.DataFrame({'a': [0, 1, 2], 'b': [5, 7, 2]})
    plot = getattr(df.plot, kind)(x='a', y='b')
    self.assertIsInstance(plot, el)