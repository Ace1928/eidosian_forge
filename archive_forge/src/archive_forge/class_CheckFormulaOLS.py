from statsmodels.compat.pandas import assert_series_equal
from io import StringIO
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import patsy
import pytest
from statsmodels.datasets import cpunish
from statsmodels.datasets.longley import load, load_pandas
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.tools.testing import assert_equal
class CheckFormulaOLS:

    @classmethod
    def setup_class(cls):
        cls.data = load()

    def test_endog_names(self):
        assert self.model.endog_names == 'TOTEMP'

    def test_exog_names(self):
        assert self.model.exog_names == ['Intercept', 'GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']

    def test_design(self):
        npt.assert_equal(self.model.exog, add_constant(self.data.exog, prepend=True))

    def test_endog(self):
        npt.assert_equal(self.model.endog, self.data.endog)

    @pytest.mark.smoke
    def test_summary(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'kurtosistest only valid for n>=20')
            self.model.fit().summary()