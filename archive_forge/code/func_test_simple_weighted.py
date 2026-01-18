import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_simple_weighted(self):
    """
        Check that weighted data raises a TypeError
        """
    estimator_list = ['fd', 'scott', 'rice', 'sturges', 'auto']
    for estimator in estimator_list:
        assert_raises(TypeError, histogram, [1, 2, 3], estimator, weights=[1, 2, 3])