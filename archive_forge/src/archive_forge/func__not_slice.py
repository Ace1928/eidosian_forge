from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def _not_slice(slices, slices_to_exclude, n):
    ind = np.array([True] * n)
    for term in slices_to_exclude:
        s = slices[term]
        ind[s] = False
    return ind