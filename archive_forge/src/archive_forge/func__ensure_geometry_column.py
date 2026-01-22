import warnings
from functools import reduce
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
def _ensure_geometry_column(df):
    """
    Helper function to ensure the geometry column is called 'geometry'.
    If another column with that name exists, it will be dropped.
    """
    if not df._geometry_column_name == 'geometry':
        if 'geometry' in df.columns:
            df.drop('geometry', axis=1, inplace=True)
        df.rename(columns={df._geometry_column_name: 'geometry'}, copy=False, inplace=True)
        df.set_geometry('geometry', inplace=True)