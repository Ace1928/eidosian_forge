import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.cross_decomposition._pls import (
from sklearn.datasets import load_linnerud, make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
@pytest.mark.parametrize('Klass', [CCA, PLSSVD, PLSRegression, PLSCanonical])
def test_pls_set_output(Klass):
    """Check `set_output` in cross_decomposition module."""
    pd = pytest.importorskip('pandas')
    X, Y = load_linnerud(return_X_y=True, as_frame=True)
    est = Klass().set_output(transform='pandas').fit(X, Y)
    X_trans, y_trans = est.transform(X, Y)
    assert isinstance(y_trans, np.ndarray)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, est.get_feature_names_out())