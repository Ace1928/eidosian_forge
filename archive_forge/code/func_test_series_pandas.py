from packaging.version import Version
from unittest import SkipTest
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import pytest
from hvplot import hvPlotTabular
from hvplot.tests.util import makeDataFrame
@series_kinds
def test_series_pandas(kind):
    ser = pd.Series(np.random.rand(10), name='A')
    ser.hvplot(kind=kind)