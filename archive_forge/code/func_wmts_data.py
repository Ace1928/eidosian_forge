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
@pytest.fixture(scope='session')
def wmts_data():
    """
    A fixture which ensures that the WMTS data is available for use in testing.
    """
    aerial = cimgt.MapQuestOpenAerial()
    tiles = [(0, 0, 0)]
    for tile in aerial.subtiles((0, 0, 0)):
        tiles.append(tile)
    for tile in tiles[1:]:
        for sub_tile in aerial.subtiles(tile):
            tiles.append(sub_tile)
    fname_template = str(_TEST_DATA_DIR / 'z_{}' / 'x_{}_y_{}.png')
    _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_version_fname = _TEST_DATA_DIR / 'version.txt'
    test_data_version = None
    try:
        with open(data_version_fname) as fh:
            test_data_version = int(fh.read().strip())
    except OSError:
        pass
    finally:
        if test_data_version != _TEST_DATA_VERSION:
            warnings.warn(f'WMTS test data is out of date, regenerating at {_TEST_DATA_DIR}.')
            shutil.rmtree(_TEST_DATA_DIR)
            _TEST_DATA_DIR.mkdir(parents=True)
            with open(data_version_fname, 'w') as fh:
                fh.write(str(_TEST_DATA_VERSION))
    for tile in tiles:
        x, y, z = tile
        fname = Path(fname_template.format(z, x, y))
        if not fname.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
            img, extent, _ = aerial.get_image(tile)
            nx, ny = (256, 256)
            x_rng = extent[1] - extent[0]
            y_rng = extent[3] - extent[2]
            pix_size_x = x_rng / nx
            pix_size_y = y_rng / ny
            upper_left_center = (extent[0] + pix_size_x / 2, extent[2] + pix_size_y / 2)
            pgw_fname = fname.with_suffix('.pgw')
            pgw_keys = {'x_pix_size': np.float64(pix_size_x), 'y_rotation': 0, 'x_rotation': 0, 'y_pix_size': np.float64(pix_size_y), 'x_center': np.float64(upper_left_center[0]), 'y_center': np.float64(upper_left_center[1])}
            _save_world(pgw_fname, pgw_keys)
            img.save(fname)