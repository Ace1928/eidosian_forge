import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
def _sample_plate_carree_scalar_field():
    return np.array([2, 4, 2, 1.2, 3, 1.2])