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
def transform_shapely(geom, crs_from, crs_to):
    from pyproj import Transformer
    if isinstance(crs_to, str):
        crs_to = ccrs.CRS(crs_to)
    if isinstance(crs_from, str):
        crs_from = ccrs.CRS(crs_from)
    project = Transformer.from_crs(crs_from, crs_to).transform
    return transform(project, geom)