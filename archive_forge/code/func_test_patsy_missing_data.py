from statsmodels.compat.pandas import assert_series_equal
from io import StringIO
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import patsy
import pytest
from statsmodels.datasets import cpunish
from statsmodels.datasets.longley import load, load_pandas
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.tools.testing import assert_equal
def test_patsy_missing_data():
    data = cpunish.load_pandas().data
    data.loc[0, 'INCOME'] = np.nan
    res = ols('EXECUTIONS ~ SOUTH + INCOME', data=data).fit()
    res2 = res.predict(data)
    assert_equal(res.fittedvalues, res2[1:])
    data = cpunish.load_pandas().data
    data.loc[0, 'INCOME'] = None
    data = data.to_records(index=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res2 = res.predict(data)
        assert 'ValueWarning' in repr(w[-1].message)
        assert 'nan values have been dropped' in repr(w[-1].message)
    assert_equal(res.fittedvalues, res2, check_index_type=False)