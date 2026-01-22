from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
class TestPickleFormula3(TestPickleFormula2):

    def setup_method(self):
        self.results = sm.OLS.from_formula('Y ~ A + B * C', data=self.data).fit()