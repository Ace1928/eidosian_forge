import warnings
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import shapely.geos
import shapely.ops
import shapely.validation
import shapely.wkb
import shapely.wkt
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
def _pygeos_to_shapely(geom):
    if geom is None:
        return None
    if compat.PYGEOS_SHAPELY_COMPAT:
        if not compat.SHAPELY_GE_20:
            geom = shapely.geos.lgeos.GEOSGeom_clone(geom._ptr)
            return shapely.geometry.base.geom_factory(geom)
    if pygeos.is_empty(geom) and pygeos.get_type_id(geom) == 0:
        return shapely.wkt.loads('POINT EMPTY')
    elif pygeos.get_type_id(geom) == 2:
        return shapely.LinearRing(shapely.wkb.loads(pygeos.to_wkb(geom)))
    else:
        return shapely.wkb.loads(pygeos.to_wkb(geom))