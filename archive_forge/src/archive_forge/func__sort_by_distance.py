import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
@classmethod
def _sort_by_distance(cls, raster, df, x, y):
    """
        Returns a dataframe of hits within a given mask around a given
        spatial location, sorted by distance from that location.
        """
    xs, ys = ([], [])
    for geom in df.geometry.array:
        gxs, gys = (geom.flat_values[::2], geom.flat_values[1::2])
        if not len(gxs):
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            xs.append((np.min(gxs) + np.max(gxs)) / 2)
            ys.append((np.min(gys) + np.max(gys)) / 2)
    dx, dy = (np.array(xs) - x, np.array(ys) - y)
    distances = pd.Series(dx * dx + dy * dy)
    return df.iloc[distances.argsort().values]