from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
class TestPickleFormula(RemoveDataPickle):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        nobs = 10000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        cls.exog = pd.DataFrame(x, columns=['A', 'B', 'C'])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)), columns=cls.exog.columns)
        cls.reduction_factor = 0.5

    def setup_method(self):
        x = self.exog
        np.random.seed(123)
        y = x.sum(1) + np.random.randn(x.shape[0])
        y = pd.Series(y, name='Y')
        X = self.exog.copy()
        X['Y'] = y
        self.results = sm.OLS.from_formula('Y ~ A + B + C', data=X).fit()