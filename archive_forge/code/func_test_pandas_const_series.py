from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange
import string
import numpy as np
from numpy.random import standard_normal
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended
def test_pandas_const_series():
    dta = longley.load_pandas()
    series = dta.exog['GNP']
    series = tools.add_constant(series, prepend=False)
    assert_string_equal('const', series.columns[1])
    assert_equal(series.var(0).iloc[1], 0)