from operator import attrgetter
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import load_iris, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, zero_one_loss
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('ClsRFE', [RFE, RFECV])
def test_pipeline_with_nans(ClsRFE):
    """Check that RFE works with pipeline that accept nans.

    Non-regression test for gh-21743.
    """
    X, y = load_iris(return_X_y=True)
    X[0, 0] = np.nan
    pipe = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
    fs = ClsRFE(estimator=pipe, importance_getter='named_steps.logisticregression.coef_')
    fs.fit(X, y)