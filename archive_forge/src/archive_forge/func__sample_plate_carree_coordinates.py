import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
def _sample_plate_carree_coordinates():
    x = np.array([-10, 0, 10, -9, 0, 9])
    y = np.array([10, 10, 10, 5, 5, 5])
    return (x, y)