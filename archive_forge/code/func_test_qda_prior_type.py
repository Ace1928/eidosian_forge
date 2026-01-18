import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import (
@pytest.mark.parametrize('priors_type', ['list', 'tuple', 'array'])
def test_qda_prior_type(priors_type):
    """Check that priors accept array-like."""
    priors = [0.5, 0.5]
    clf = QuadraticDiscriminantAnalysis(priors=_convert_container([0.5, 0.5], priors_type)).fit(X6, y6)
    assert isinstance(clf.priors_, np.ndarray)
    assert_array_equal(clf.priors_, priors)