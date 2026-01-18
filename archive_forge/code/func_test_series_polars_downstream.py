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
def test_series_polars_downstream(kind):
    if Version(pl.__version__) < Version('0.20.3'):
        raise SkipTest('plot namespace in Polars introduced in 0.20.3')
    ser = pl.Series(values=np.random.rand(10), name='A')
    assert isinstance(ser, pl.Series)
    ser.plot(kind=kind)