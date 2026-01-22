import warnings
from functools import reduce
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
def _overlay_symmetric_diff(df1, df2):
    """
    Overlay Symmetric Difference operation used in overlay function
    """
    dfdiff1 = _overlay_difference(df1, df2)
    dfdiff2 = _overlay_difference(df2, df1)
    dfdiff1['__idx1'] = range(len(dfdiff1))
    dfdiff2['__idx2'] = range(len(dfdiff2))
    dfdiff1['__idx2'] = np.nan
    dfdiff2['__idx1'] = np.nan
    _ensure_geometry_column(dfdiff1)
    _ensure_geometry_column(dfdiff2)
    dfsym = dfdiff1.merge(dfdiff2, on=['__idx1', '__idx2'], how='outer', suffixes=('_1', '_2'))
    geometry = dfsym.geometry_1.copy()
    geometry.name = 'geometry'
    geometry.loc[dfsym.geometry_1.isnull()] = dfsym.loc[dfsym.geometry_1.isnull(), 'geometry_2']
    dfsym.drop(['geometry_1', 'geometry_2'], axis=1, inplace=True)
    dfsym.reset_index(drop=True, inplace=True)
    dfsym = GeoDataFrame(dfsym, geometry=geometry, crs=df1.crs)
    return dfsym