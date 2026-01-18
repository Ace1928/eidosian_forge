import io
from pathlib import Path
import pickle
import shutil
import sys
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from PIL import Image
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.io.img_nest as cimg_nest
import cartopy.io.img_tiles as cimgt
@pytest.mark.parametrize('fname, expected', [('one', ['one.w', 'one.W', 'ONE.w', 'ONE.W']), ('one.png', ['one.pngw', 'one.pgw', 'one.PNGW', 'one.PGW', 'ONE.pngw', 'ONE.pgw', 'ONE.PNGW', 'ONE.PGW']), ('/one.png', ['/one.pngw', '/one.pgw', '/one.PNGW', '/one.PGW', '/ONE.pngw', '/ONE.pgw', '/ONE.PNGW', '/ONE.PGW']), ('/one/two.png', ['/one/two.pngw', '/one/two.pgw', '/one/two.PNGW', '/one/two.PGW', '/one/TWO.pngw', '/one/TWO.pgw', '/one/TWO.PNGW', '/one/TWO.PGW']), ('/one/two/THREE.png', ['/one/two/THREE.pngw', '/one/two/THREE.pgw', '/one/two/THREE.PNGW', '/one/two/THREE.PGW', '/one/two/three.pngw', '/one/two/three.pgw', '/one/two/three.PNGW', '/one/two/three.PGW'])])
def test_world_files(fname, expected):
    if sys.platform == 'win32':
        fname = fname.replace('/', '\\')
        expected = [f.replace('/', '\\') for f in expected]
        if fname.startswith('\\'):
            fname = 'c:' + fname
            expected = ['c:' + f for f in expected]
    assert cimg_nest.Img.world_files(fname) == expected