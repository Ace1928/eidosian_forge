import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
def _generate_spherical_points(ndim=3, n_pts=2):
    np.random.seed(123)
    points = np.random.normal(size=(n_pts, ndim))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return (points[0], points[1])