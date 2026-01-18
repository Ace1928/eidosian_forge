import numpy as np
from numpy.lib.histograms import histogram, histogramdd, histogram_bin_edges
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
import pytest
def test_outlier(self):
    """
        Check the FD, Scott and Doane with outliers.

        The FD estimates a smaller binwidth since it's less affected by
        outliers. Since the range is so (artificially) large, this means more
        bins, most of which will be empty, but the data of interest usually is
        unaffected. The Scott estimator is more affected and returns fewer bins,
        despite most of the variance being in one area of the data. The Doane
        estimator lies somewhere between the other two.
        """
    xcenter = np.linspace(-10, 10, 50)
    outlier_dataset = np.hstack((np.linspace(-110, -100, 5), xcenter))
    outlier_resultdict = {'fd': 21, 'scott': 5, 'doane': 11, 'stone': 6}
    for estimator, numbins in outlier_resultdict.items():
        a, b = np.histogram(outlier_dataset, estimator)
        assert_equal(len(a), numbins)