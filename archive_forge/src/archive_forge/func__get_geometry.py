import json
import warnings
import numpy as np
import pandas as pd
import shapely.errors
from pandas import DataFrame, Series
from pandas.core.accessor import CachedAccessor
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from pyproj import CRS
from geopandas.array import GeometryArray, GeometryDtype, from_shapely, to_wkb, to_wkt
from geopandas.base import GeoPandasBase, is_geometry_type
from geopandas.geoseries import GeoSeries
import geopandas.io
from geopandas.explore import _explore
from . import _compat as compat
from ._decorator import doc
def _get_geometry(self):
    if self._geometry_column_name not in self:
        if self._geometry_column_name is None:
            msg = 'You are calling a geospatial method on the GeoDataFrame, but the active geometry column to use has not been set. '
        else:
            msg = f"You are calling a geospatial method on the GeoDataFrame, but the active geometry column ('{self._geometry_column_name}') is not present. "
        geo_cols = list(self.columns[self.dtypes == 'geometry'])
        if len(geo_cols) > 0:
            msg += f'\nThere are columns with geometry data type ({geo_cols}), and you can either set one as the active geometry with df.set_geometry("name") or access the column as a GeoSeries (df["name"]) and call the method directly on it.'
        else:
            msg += '\nThere are no existing columns with geometry data type. You can add a geometry column as the active geometry column with df.set_geometry. '
        raise AttributeError(msg)
    return self[self._geometry_column_name]