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
def test_perfect_prediction():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    iris_dir = os.path.join(cur_dir, '..', '..', 'genmod', 'tests', 'results')
    iris_dir = os.path.abspath(iris_dir)
    iris = np.genfromtxt(os.path.join(iris_dir, 'iris.csv'), delimiter=',', skip_header=1)
    y = iris[:, -1]
    X = iris[:, :-1]
    X = X[y != 2]
    y = y[y != 2]
    X = sm.add_constant(X, prepend=True)
    mod = Logit(y, X)
    mod.raise_on_perfect_prediction = True
    assert_raises(PerfectSeparationError, mod.fit, maxiter=1000)
    mod.raise_on_perfect_prediction = False
    with pytest.warns(ConvergenceWarning):
        res = mod.fit(disp=False, maxiter=50)
    assert_(not res.mle_retvals['converged'])
    mod.fit(method='bfgs', disp=False, maxiter=50)