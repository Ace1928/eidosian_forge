import sys
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from holoviews.core.util import isscalar, unique_iterator, unique_array
from holoviews.core.data import Dataset, Interface, MultiInterface, PandasAPI
from holoviews.core.data.interface import DataError
from holoviews.core.data import PandasInterface
from holoviews.core.data.spatialpandas import get_value_array
from holoviews.core.dimension import dimension_name
from holoviews.element import Path
from ..util import asarray, geom_to_array, geom_types, geom_length
from .geom_dict import geom_from_dict
def to_geopandas(data, xdim, ydim, columns=None, geom='point'):
    """Converts list of dictionary format geometries to spatialpandas line geometries.

    Args:
        data: List of dictionaries representing individual geometries
        xdim: Name of x-coordinates column
        ydim: Name of y-coordinates column
        ring: Whether the data represents a closed ring

    Returns:
        A spatialpandas.GeoDataFrame version of the data
    """
    from geopandas import GeoDataFrame
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiPolygon, MultiLineString
    if columns is None:
        columns = []
    poly = any(('holes' in d for d in data)) or geom == 'Polygon'
    if poly:
        single_type, multi_type = (Polygon, MultiPolygon)
    elif geom == 'Line':
        single_type, multi_type = (LineString, MultiLineString)
    else:
        single_type, multi_type = (Point, MultiPoint)
    converted = defaultdict(list)
    for geom_dict in data:
        geom_dict = dict(geom_dict)
        geom = geom_from_dict(geom_dict, xdim, ydim, single_type, multi_type)
        for c, v in geom_dict.items():
            converted[c].append(v)
        converted['geometry'].append(geom)
    return GeoDataFrame(converted, columns=['geometry'] + columns)