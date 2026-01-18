import hashlib
import os
import types
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_arr_almost
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

    Tests that setting the Ordnance Survey tile style works as expected.

    This is essentially just assures information is properly propagated through
    the class structure.
    