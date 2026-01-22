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
def _target_extents(extent, requested_projection, available_projection):
    """
    Translate the requested extent in the display projection into a list of
    extents in the projection available from the service (multiple if it
    crosses seams).

    The extents are represented as (min_x, max_x, min_y, max_y).

    """
    min_x, max_x, min_y, max_y = extent
    target_box = sgeom.box(min_x, min_y, max_x, max_y)
    buffered_target_box = target_box.buffer(requested_projection.threshold, resolution=1)
    fudge_mode = buffered_target_box.contains(requested_projection.domain)
    if fudge_mode:
        target_box = requested_projection.domain.buffer(-requested_projection.threshold)
    polys = available_projection.project_geometry(target_box, requested_projection)
    target_extents = []
    for poly in polys.geoms:
        min_x, min_y, max_x, max_y = poly.bounds
        if fudge_mode:
            radius = min(max_x - min_x, max_y - min_y) / 5.0
            radius = min(radius, available_projection.threshold * 15)
            poly = poly.buffer(radius, resolution=1)
            poly = available_projection.domain.intersection(poly)
            min_x, min_y, max_x, max_y = poly.bounds
        target_extents.append((min_x, max_x, min_y, max_y))
    return target_extents