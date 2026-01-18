import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
def test_unknown_fa_method_error():
    mod = Factor(X.iloc[:, 1:-1], 2, method='ab')
    assert_raises(ValueError, mod.fit)