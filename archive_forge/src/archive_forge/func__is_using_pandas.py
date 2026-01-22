from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def _is_using_pandas(endog, exog):
    from statsmodels.compat.pandas import data_klasses as klasses
    return isinstance(endog, klasses) or isinstance(exog, klasses)