from packaging.version import Version
from unittest import SkipTest
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import pytest
from hvplot import hvPlotTabular
from hvplot.tests.util import makeDataFrame
@pytest.mark.skipif(skip_polar, reason='polars not installed')
@series_kinds
def test_series_polars(kind):
    ser = pl.Series(values=np.random.rand(10), name='A')
    assert isinstance(ser, pl.Series)
    ser.hvplot(kind=kind)