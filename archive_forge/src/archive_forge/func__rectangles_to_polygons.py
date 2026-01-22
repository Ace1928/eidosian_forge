from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from .geom import geom
from .geom_polygon import geom_polygon
def _rectangles_to_polygons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert rect data to polygons

    Paramters
    ---------
    df : dataframe
        Dataframe with *xmin*, *xmax*, *ymin* and *ymax* columns,
        plus others for aesthetics ...

    Returns
    -------
    data : dataframe
        Dataframe with *x* and *y* columns, plus others for
        aesthetics ...
    """
    n = len(df)
    xmin_idx = np.tile([True, True, False, False], n)
    xmax_idx = ~xmin_idx
    ymin_idx = np.tile([True, False, False, True], n)
    ymax_idx = ~ymin_idx
    x = np.empty(n * 4)
    y = np.empty(n * 4)
    x[xmin_idx] = df['xmin'].repeat(2)
    x[xmax_idx] = df['xmax'].repeat(2)
    y[ymin_idx] = df['ymin'].repeat(2)
    y[ymax_idx] = df['ymax'].repeat(2)
    other_cols = df.columns.difference(['x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'])
    d = {str(col): np.repeat(df[col].to_numpy(), 4) for col in other_cols}
    data = pd.DataFrame({'x': x, 'y': y, **d})
    return data