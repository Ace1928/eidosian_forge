from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, _fit_context
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from sklearn.utils._param_validation import (
def test_pandas_na_constraint_with_pd_na():
    """Add a specific test for checking support for `pandas.NA`."""
    pd = pytest.importorskip('pandas')
    na_constraint = _PandasNAConstraint()
    assert na_constraint.is_satisfied_by(pd.NA)
    assert not na_constraint.is_satisfied_by(np.array([1, 2, 3]))