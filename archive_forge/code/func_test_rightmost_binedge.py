import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_rightmost_binedge(self):
    x = [0.9999999995]
    bins = [[0.0, 0.5, 1.0]]
    hist, _ = histogramdd(x, bins=bins)
    assert_(hist[0] == 0.0)
    assert_(hist[1] == 1.0)
    x = [1.0]
    bins = [[0.0, 0.5, 1.0]]
    hist, _ = histogramdd(x, bins=bins)
    assert_(hist[0] == 0.0)
    assert_(hist[1] == 1.0)
    x = [1.0000000001]
    bins = [[0.0, 0.5, 1.0]]
    hist, _ = histogramdd(x, bins=bins)
    assert_(hist[0] == 0.0)
    assert_(hist[1] == 0.0)
    x = [1.0001]
    bins = [[0.0, 0.5, 1.0]]
    hist, _ = histogramdd(x, bins=bins)
    assert_(hist[0] == 0.0)
    assert_(hist[1] == 0.0)