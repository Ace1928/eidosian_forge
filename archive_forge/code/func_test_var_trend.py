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
def test_var_trend():
    data = get_macrodata().view((float, 3), type=np.ndarray)
    model = VAR(data)
    results = model.fit(4)
    irf = results.irf(10)
    data_nc = data - data.mean(0)
    model_nc = VAR(data_nc)
    results_nc = model_nc.fit(4, trend='n')
    with pytest.raises(ValueError):
        model.fit(4, trend='t')