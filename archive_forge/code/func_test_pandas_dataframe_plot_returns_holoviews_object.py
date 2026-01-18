from unittest import TestCase, SkipTest
import holoviews as hv
import pandas as pd
import pytest
from packaging.version import Version
from parameterized import parameterized
from hvplot.converter import HoloViewsConverter
from hvplot.plotting import plot
from hvplot.tests.util import makeDataFrame
@parameterized.expand(no_args_mapping)
def test_pandas_dataframe_plot_returns_holoviews_object(self, kind, el):
    df = pd.DataFrame([0, 1, 2])
    plot = getattr(df.plot, kind)()
    self.assertIsInstance(plot, el)