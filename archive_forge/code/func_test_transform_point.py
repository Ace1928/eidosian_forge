from copy import deepcopy
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_transform_point(self, plate_carree):
    src_expected = ((self.point_a_plate_carree, self.expected_a), (self.point_b_plate_carree, self.expected_b))
    for src, expected in src_expected:
        res = self.oblique_mercator.transform_point(*src, src_crs=plate_carree)
        np.testing.assert_array_almost_equal(res, expected, decimal=4)