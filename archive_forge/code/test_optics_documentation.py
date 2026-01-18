import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
Check that cluster correction using predecessor is working as expected.

    In the following example, the predecessor correction was not working properly
    since it was not using the right indices.

    This non-regression test check that reordering the data does not change the results.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26324
    