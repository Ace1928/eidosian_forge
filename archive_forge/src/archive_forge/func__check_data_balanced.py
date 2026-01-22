from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def _check_data_balanced(self):
    """raise if data is not balanced

        This raises a ValueError if the data is not balanced, and
        returns None if it is balance

        Return might change
        """
    factor_levels = 1
    for wi in self.within:
        factor_levels *= len(self.data[wi].unique())
    cell_count = {}
    for index in range(self.data.shape[0]):
        key = []
        for col in self.within:
            key.append(self.data[col].iloc[index])
        key = tuple(key)
        if key in cell_count:
            cell_count[key] = cell_count[key] + 1
        else:
            cell_count[key] = 1
    error_message = 'Data is unbalanced.'
    if len(cell_count) != factor_levels:
        raise ValueError(error_message)
    count = cell_count[key]
    for key in cell_count:
        if count != cell_count[key]:
            raise ValueError(error_message)
    if self.data.shape[0] > count * factor_levels:
        raise ValueError('There are more than 1 element in a cell! Missing factors?')