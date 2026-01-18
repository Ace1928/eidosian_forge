from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
def test_predict_missing(self):
    ex = self.data[:5].copy()
    ex.iloc[0, 1] = np.nan
    predicted1 = self.res.predict(ex)
    predicted2 = self.res.predict(ex[1:])
    assert_index_equal(predicted1.index, ex.index)
    assert_series_equal(predicted1.iloc[1:], predicted2)
    assert_equal(predicted1.values[0], np.nan)