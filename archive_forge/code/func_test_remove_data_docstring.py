from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
def test_remove_data_docstring(self):
    assert self.results.remove_data.__doc__ is not None