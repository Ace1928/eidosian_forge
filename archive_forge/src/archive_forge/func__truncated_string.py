import warnings
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from geopandas import _vectorized
def _truncated_string(geom):
    """Truncated WKT repr of geom"""
    s = str(geom)
    if len(s) > 100:
        return s[:100] + '...'
    else:
        return s