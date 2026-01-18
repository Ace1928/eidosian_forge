import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
@pytest.mark.parametrize('bins', ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges'])
def test_signed_integer_data(self, bins):
    a = np.array([-2, 0, 127], dtype=np.int8)
    hist, edges = np.histogram(a, bins=bins)
    hist32, edges32 = np.histogram(a.astype(np.int32), bins=bins)
    assert_array_equal(hist, hist32)
    assert_array_equal(edges, edges32)