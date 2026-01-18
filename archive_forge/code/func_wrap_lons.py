import numpy as np
import shapely
import shapely.geometry as sgeom
from cartopy import crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, QuadtreeTiles
from holoviews.element import Tiles
from packaging.version import Version
from shapely.geometry import (
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import transform
from ._warnings import warn
def wrap_lons(lons, base, period):
    """
    Wrap longitude values into the range between base and base+period.
    """
    lons = lons.astype(np.float64)
    return (lons - base + period * 2) % period + base