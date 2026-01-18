import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_region_types(self):
    sv = SphericalVoronoi(self.points)
    dtype = type(sv.regions[0][0])
    for region in sv.regions:
        assert isinstance(region, list)
    sv.sort_vertices_of_regions()
    assert type(sv.regions[0][0]) == dtype
    sv.sort_vertices_of_regions()
    assert type(sv.regions[0][0]) == dtype