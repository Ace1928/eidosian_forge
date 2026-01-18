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
def test_var_constant():
    import datetime
    from pandas import DataFrame, DatetimeIndex
    series = np.array([[2.0, 2.0], [1, 2.0], [1, 2.0], [1, 2.0], [1.0, 2.0]])
    data = DataFrame(series)
    d = datetime.datetime.now()
    delta = datetime.timedelta(days=1)
    index = []
    for i in range(data.shape[0]):
        index.append(d)
        d += delta
    data.index = DatetimeIndex(index)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ValueWarning)
        model = VAR(data)
    with pytest.raises(ValueError):
        model.fit(1)