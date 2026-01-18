import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_old_radius_api_error(self):
    with pytest.raises(ValueError, match='`radius` is `None`. *'):
        SphericalVoronoi(self.points, radius=None)