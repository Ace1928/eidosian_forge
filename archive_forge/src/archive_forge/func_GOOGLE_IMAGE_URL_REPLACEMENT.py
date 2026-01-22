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
def GOOGLE_IMAGE_URL_REPLACEMENT(self, tile):
    pytest.xfail(reason='Google has deprecated the tile API used in this test')
    x, y, z = tile
    return f'https://chart.googleapis.com/chart?chst=d_text_outline&chs=256x256&chf=bg,s,00000055&chld=FFFFFF|16|h|000000|b||||Google:%20%20({x},{y})|Zoom%20{z}||||||____________________________'