from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
def test_var_maxlag(reset_randomstate):
    y = np.random.standard_normal((22, 10))
    VAR(y).fit(maxlags=None, ic='aic')
    with pytest.raises(ValueError, match='maxlags is too large'):
        VAR(y).fit(maxlags=8, ic='aic')