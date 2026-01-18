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
def test_quadtree_wts():
    qt = cimgt.QuadtreeTiles()
    ll_target_domain = sgeom.box(-15, 50, 0, 60)
    multi_poly = qt.crs.project_geometry(ll_target_domain, ccrs.PlateCarree())
    target_domain = multi_poly.geoms[0]
    with pytest.raises(ValueError):
        list(qt.find_images(target_domain, 0))
    assert qt.tms_to_quadkey((1, 1, 1)) == '1'
    assert qt.quadkey_to_tms('1') == (1, 1, 1)
    assert qt.tms_to_quadkey((8, 9, 4)) == '1220'
    assert qt.quadkey_to_tms('1220') == (8, 9, 4)
    assert tuple(qt.find_images(target_domain, 1)) == ('0', '1')
    assert tuple(qt.find_images(target_domain, 2)) == ('03', '12')
    assert list(qt.subtiles('0')) == ['00', '01', '02', '03']
    assert list(qt.subtiles('11')) == ['110', '111', '112', '113']
    with pytest.raises(ValueError):
        qt.tileextent('4')
    assert_arr_almost(qt.tileextent(''), KNOWN_EXTENTS[0, 0, 0])
    assert_arr_almost(qt.tileextent(qt.tms_to_quadkey((2, 0, 2), google=True)), KNOWN_EXTENTS[2, 0, 2])
    assert_arr_almost(qt.tileextent(qt.tms_to_quadkey((0, 2, 2), google=True)), KNOWN_EXTENTS[0, 2, 2])
    assert_arr_almost(qt.tileextent(qt.tms_to_quadkey((2, 0, 2), google=True)), KNOWN_EXTENTS[2, 0, 2])
    assert_arr_almost(qt.tileextent(qt.tms_to_quadkey((2, 2, 2), google=True)), KNOWN_EXTENTS[2, 2, 2])
    assert_arr_almost(qt.tileextent(qt.tms_to_quadkey((8, 9, 4), google=True)), KNOWN_EXTENTS[8, 9, 4])