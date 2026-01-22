import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def _choose_matrix(self, tile_matrices, meters_per_unit, max_pixel_span):
    tile_matrices = sorted(tile_matrices, key=lambda tm: tm.scaledenominator, reverse=True)
    max_scale = max_pixel_span * meters_per_unit / METERS_PER_PIXEL
    for tm in tile_matrices:
        if tm.scaledenominator <= max_scale:
            return tm
    return tile_matrices[-1]