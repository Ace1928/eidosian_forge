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
def _shapely_normalize(geom):
    """
    Small helper function for now because it is not yet available in Shapely.
    """
    from ctypes import c_int, c_void_p
    from shapely.geometry.base import geom_factory
    from shapely.geos import lgeos
    lgeos._lgeos.GEOSNormalize_r.restype = c_int
    lgeos._lgeos.GEOSNormalize_r.argtypes = [c_void_p, c_void_p]
    geom_cloned = lgeos.GEOSGeom_clone(geom._geom)
    lgeos._lgeos.GEOSNormalize_r(lgeos.geos_handle, geom_cloned)
    return geom_factory(geom_cloned)