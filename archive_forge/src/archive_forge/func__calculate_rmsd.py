import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
def _calculate_rmsd(P, Q):
    """Calculates the root-mean-square distance between the points of P and Q.
    The distance is taken as the minimum over all possible matchings. It is
    zero if P and Q are identical and non-zero if not.
    """
    distance_matrix = cdist(P, Q, metric='sqeuclidean')
    matching = linear_sum_assignment(distance_matrix)
    return np.sqrt(distance_matrix[matching].sum())