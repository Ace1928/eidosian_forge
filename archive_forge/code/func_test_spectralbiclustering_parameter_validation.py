import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('params, type_err, err_msg', [({'n_clusters': 6}, ValueError, 'n_clusters should be <= n_samples=5'), ({'n_clusters': (3, 3, 3)}, ValueError, 'Incorrect parameter n_clusters'), ({'n_clusters': (3, 6)}, ValueError, 'Incorrect parameter n_clusters'), ({'n_components': 3, 'n_best': 4}, ValueError, 'n_best=4 must be <= n_components=3')])
def test_spectralbiclustering_parameter_validation(params, type_err, err_msg):
    """Check parameters validation in `SpectralBiClustering`"""
    data = np.arange(25).reshape((5, 5))
    model = SpectralBiclustering(**params)
    with pytest.raises(type_err, match=err_msg):
        model.fit(data)