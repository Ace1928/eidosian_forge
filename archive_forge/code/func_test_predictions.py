import re
import numpy as np
import pytest
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('use_feature_names', (True, False))
def test_predictions(global_random_seed, use_feature_names):
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    f_0 = rng.rand(n_samples)
    f_1 = rng.rand(n_samples)
    X = np.c_[f_0, f_1]
    columns_name = ['f_0', 'f_1']
    constructor_name = 'dataframe' if use_feature_names else 'array'
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)
    y = 5 * f_0 + np.sin(10 * np.pi * f_0) - 5 * f_1 - np.cos(10 * np.pi * f_1) + noise
    if use_feature_names:
        monotonic_cst = {'f_0': +1, 'f_1': -1}
    else:
        monotonic_cst = [+1, -1]
    gbdt = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst)
    gbdt.fit(X, y)
    linspace = np.linspace(0, 1, 100)
    sin = np.sin(linspace)
    constant = np.full_like(linspace, fill_value=0.5)
    X = np.c_[linspace, constant]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert is_increasing(pred)
    X = np.c_[sin, constant]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert np.all((np.diff(pred) >= 0) == (np.diff(sin) >= 0))
    X = np.c_[constant, linspace]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert is_decreasing(pred)
    X = np.c_[constant, sin]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert ((np.diff(pred) <= 0) == (np.diff(sin) >= 0)).all()