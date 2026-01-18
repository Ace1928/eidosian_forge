import types
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.tests.test_img_tiles as ctest_tiles
@pytest.mark.mpl_image_compare(filename='imshow_natural_earth_ortho.png')
def test_pil_Image():
    img = Image.open(NATURAL_EARTH_IMG)
    source_proj = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.Orthographic())
    ax.imshow(img, transform=source_proj, extent=[-180, 180, -90, 90])
    return ax.figure