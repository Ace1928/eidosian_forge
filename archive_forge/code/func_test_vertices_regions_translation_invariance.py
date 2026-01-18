import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_vertices_regions_translation_invariance(self):
    sv_origin = SphericalVoronoi(self.points)
    center = np.array([1, 1, 1])
    sv_translated = SphericalVoronoi(self.points + center, center=center)
    assert_equal(sv_origin.regions, sv_translated.regions)
    assert_array_almost_equal(sv_origin.vertices + center, sv_translated.vertices)