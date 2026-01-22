from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def _is_using_ndarray_type(endog, exog):
    return type(endog) is np.ndarray and (type(exog) is np.ndarray or exog is None)