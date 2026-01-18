import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
def test_transform_dense():
    sel = StepSelector()
    Xt_actual = sel.fit(X, y).transform(X)
    Xt_actual2 = StepSelector().fit_transform(X, y)
    assert_array_equal(Xt, Xt_actual)
    assert_array_equal(Xt, Xt_actual2)
    assert np.int32 == sel.transform(X.astype(np.int32)).dtype
    assert np.float32 == sel.transform(X.astype(np.float32)).dtype
    names_t_actual = sel.transform([feature_names])
    assert_array_equal(feature_names_t, names_t_actual.ravel())
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))