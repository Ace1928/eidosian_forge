from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def _get_covariance(model, robust):
    if robust is None:
        return model.cov_params()
    elif robust == 'hc0':
        return model.cov_HC0
    elif robust == 'hc1':
        return model.cov_HC1
    elif robust == 'hc2':
        return model.cov_HC2
    elif robust == 'hc3':
        return model.cov_HC3
    else:
        raise ValueError('robust options %s not understood' % robust)