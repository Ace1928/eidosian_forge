import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_summary_col_ordering_preserved(self):
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
    x2 = pd.DataFrame(x2, columns=['const', 'b', 'a'])
    y1 = [6, 4, 2, 7, 4]
    y2 = [8, 5, 0, 12, 4]
    reg1 = OLS(y1, x2).fit()
    reg2 = OLS(y2, x2).fit()
    info_dict = {'R2': lambda x: f'{int(x.rsquared):.3f}', 'N': lambda x: f'{int(x.nobs):d}'}
    original = actual = summary_col([reg1, reg2], float_format='%0.4f')
    actual = summary_col([reg1, reg2], regressor_order=['a', 'b'], float_format='%0.4f', info_dict=info_dict)
    variables = ('const', 'b', 'a')
    for line in str(original).split('\n'):
        for variable in variables:
            if line.startswith(variable):
                assert line in str(actual)