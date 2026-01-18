from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
@pytest.mark.smoke
def test_raise_no_missing(self):
    sm_data.handle_data(pd.Series(np.random.random(20)), pd.DataFrame(np.random.random((20, 2))), 'raise')