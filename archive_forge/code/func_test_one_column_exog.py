import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_one_column_exog(self, close_figures):
    from statsmodels.formula.api import ols
    res = ols('y~var1-1', data=self.data).fit()
    plot_regress_exog(res, 'var1')
    res = ols('y~var1', data=self.data).fit()
    plot_regress_exog(res, 'var1')