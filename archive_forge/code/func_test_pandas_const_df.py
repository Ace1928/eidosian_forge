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
def test_pandas_const_df():
    dta = longley.load_pandas().exog
    dta = tools.add_constant(dta, prepend=False)
    assert_string_equal('const', dta.columns[-1])
    assert_equal(dta.var(0).iloc[-1], 0)