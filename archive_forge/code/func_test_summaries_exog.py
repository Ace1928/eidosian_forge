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
def test_summaries_exog(reset_randomstate):
    y = np.random.standard_normal((500, 6))
    df = pd.DataFrame(y)
    cols = [f'endog_{i}' for i in range(2)] + [f'exog_{i}' for i in range(4)]
    df.columns = cols
    df.index = pd.date_range('1-1-1950', periods=500, freq='MS')
    endog = df.iloc[:, :2]
    exog = df.iloc[:, 2:]
    res = VAR(endog=endog, exog=exog).fit(maxlags=0)
    summ = res.summary().summary
    assert 'exog_0' in summ
    assert 'exog_1' in summ
    assert 'exog_2' in summ
    assert 'exog_3' in summ
    res = VAR(endog=endog, exog=exog).fit(maxlags=2)
    summ = res.summary().summary
    assert 'exog_0' in summ
    assert 'exog_1' in summ
    assert 'exog_2' in summ
    assert 'exog_3' in summ