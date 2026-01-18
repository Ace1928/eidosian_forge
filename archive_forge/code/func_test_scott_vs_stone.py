import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_scott_vs_stone(self):
    """Verify that Scott's rule and Stone's rule converges for normally distributed data"""

    def nbins_ratio(seed, size):
        rng = np.random.RandomState(seed)
        x = rng.normal(loc=0, scale=2, size=size)
        a, b = (len(np.histogram(x, 'stone')[0]), len(np.histogram(x, 'scott')[0]))
        return a / (a + b)
    ll = [[nbins_ratio(seed, size) for size in np.geomspace(start=10, stop=100, num=4).round().astype(int)] for seed in range(10)]
    avg = abs(np.mean(ll, axis=0) - 0.5)
    assert_almost_equal(avg, [0.15, 0.09, 0.08, 0.03], decimal=2)