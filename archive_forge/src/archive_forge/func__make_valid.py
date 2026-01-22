import warnings
from functools import reduce
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
def _make_valid(df):
    df = df.copy()
    if df.geom_type.isin(polys).all():
        mask = ~df.geometry.is_valid
        col = df._geometry_column_name
        if make_valid:
            df.loc[mask, col] = df.loc[mask, col].buffer(0)
        elif mask.any():
            raise ValueError(f'You have passed make_valid=False along with {mask.sum()} invalid input geometries. Use make_valid=True or make sure that all geometries are valid before using overlay.')
    return df