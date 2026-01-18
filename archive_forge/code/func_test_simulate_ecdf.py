import pytest
import numpy as np
import scipy.stats
from ...stats.ecdf_utils import (
@pytest.mark.parametrize('dist', [scipy.stats.norm(3, 10), scipy.stats.binom(10, 0.5)], ids=['continuous', 'discrete'])
@pytest.mark.parametrize('seed', [32, 87])
def test_simulate_ecdf(dist, seed):
    """Test _simulate_ecdf."""
    ndraws = 1000
    eval_points = np.arange(0, 1, 0.1)
    rvs = dist.rvs
    random_state = np.random.default_rng(seed)
    ecdf = _simulate_ecdf(ndraws, eval_points, rvs, random_state=random_state)
    random_state = np.random.default_rng(seed)
    ecdf_expected = compute_ecdf(np.sort(rvs(ndraws, random_state=random_state)), eval_points)
    assert np.allclose(ecdf, ecdf_expected)