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
def _tile_from_img(img):
    """
    Turns an img into the appropriate x, y, z tile based on its filename.

    Imgs have a filename attribute which is something
    like "lib/cartopy/data/wmts/aerial/z_0/x_0_y0.png"

    """
    _, z = img.filename.parent.name.split('_')
    _, x, _, y = img.filename.stem.split('_')
    return (int(x), int(y), int(z))