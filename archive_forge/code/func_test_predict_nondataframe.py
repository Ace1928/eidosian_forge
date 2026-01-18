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
def test_predict_nondataframe():
    df = pd.DataFrame([[3, 0.03], [10, 0.06], [20, 0.12]], columns=['BSA', 'Absorbance'])
    model = ols('Absorbance ~ BSA', data=df)
    fit = model.fit()
    error = patsy.PatsyError
    with pytest.raises(error):
        fit.predict([0.25])