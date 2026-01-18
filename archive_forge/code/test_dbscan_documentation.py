import pickle
import warnings
import numpy as np
import pytest
from scipy.spatial import distance
from sklearn.cluster import DBSCAN, dbscan
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
Check that we don't modify in-place the pre-computed sparse matrix.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27508
    