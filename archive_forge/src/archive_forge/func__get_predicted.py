import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def _get_predicted(self, obj):
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, pd.Series):
        return obj.values
    elif hasattr(obj, 'predicted_values'):
        return obj.predicted_values
    else:
        raise ValueError('cannot obtain predicted values from %s' % obj.__class__)