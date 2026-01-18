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
def test_var_cov_params_pandas(bivariate_var_data):
    df = pd.DataFrame(bivariate_var_data, columns=['x', 'y'])
    mod = VAR(df)
    res = mod.fit(2)
    cov = res.cov_params()
    assert isinstance(cov, pd.DataFrame)
    exog_names = ('const', 'L1.x', 'L1.y', 'L2.x', 'L2.y')
    index = pd.MultiIndex.from_product((exog_names, ('x', 'y')))
    assert_index_equal(cov.index, cov.columns)
    assert_index_equal(cov.index, index)