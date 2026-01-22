import param
import numpy as np
from bokeh.models import MercatorTileSource
from cartopy import crs as ccrs
from cartopy.feature import Feature as cFeature
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.shapereader import Reader
from holoviews.core import Element2D, Dimension, Dataset as HvDataset, NdOverlay, Overlay
from holoviews.core import util
from holoviews.element import (
from holoviews.element.selection import Selection2DExpr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import (
from shapely.ops import unary_union
from ..util import (
class Polygons(_Element, HvPolygons):
    """
    Polygons is a Path Element type that may contain any number of
    closed paths with an associated value and a coordinate reference
    system.
    """
    group = param.String(default='Polygons', constant=True)

    def geom(self, union=False, projection=None):
        """
        Converts the Path to a shapely geometry.

        Parameters
        ----------
        union: boolean (default=False)
            Whether to compute a union between the geometries
        projection : EPSG string | Cartopy CRS | None
            Whether to project the geometry to other coordinate system

        Returns
        -------
        A shapely geometry
        """
        geoms = expand_geoms([g['geometry'] for g in polygons_to_geom_dicts(self)])
        ngeoms = len(geoms)
        if not ngeoms:
            geom = GeometryCollection()
        elif ngeoms == 1:
            geom = geoms[0]
        else:
            geom = MultiPolygon(geoms)
        if projection:
            geom = transform_shapely(geom, self.crs, projection)
        return unary_union(geom) if union else geom