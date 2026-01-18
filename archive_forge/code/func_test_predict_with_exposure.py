from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_predict_with_exposure():
    import pandas as pd
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, 4], 'constant': [1] * 4, 'exposure': [np.exp(1)] * 4, 'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)
    mod1 = CountModel.from_formula('Foo ~ Bar', data=df, exposure=df['exposure'])
    params = np.array([1, 0.4])
    pred = mod1.predict(params, which='linear')
    X = df[['constant', 'Bar']].values
    expected = np.dot(X, params) + 1
    assert_allclose(pred, expected)
    pred2 = mod1.predict(params, exposure=[np.exp(2)] * 4, which='linear')
    expected2 = expected + 1
    assert_allclose(pred2, expected2)