import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def true_decision_function(input_features):
    """Ground truth decision function

        This is a very simple yet asymmetric decision tree. Therefore the
        grower code should have no trouble recovering the decision function
        from 10000 training samples.
        """
    if input_features[0] <= n_bins // 2:
        return -1
    else:
        return -1 if input_features[1] <= n_bins // 3 else 1