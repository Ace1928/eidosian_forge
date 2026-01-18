import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning

        R code:
        ------

        > data("quakes")
        > x = quakes[1:50, 1:3]
        > y = quakes[51:100, 1:3]
        > dcov.test(x, y, R=200)

            dCov independence test (permutation test)

        data:  index 1, replicates 200
        nV^2 = 45046, p-value = 0.4577
        sample estimates:
            dCov
        30.01526
        