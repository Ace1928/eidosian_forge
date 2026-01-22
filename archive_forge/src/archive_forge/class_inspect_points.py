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
class inspect_points(inspect_base):

    @classmethod
    def _element(cls, raster, df):
        return Points(df, kdims=raster.kdims, vdims=cls._vdims(raster, df))

    @classmethod
    def _sort_by_distance(cls, raster, df, x, y):
        """
        Returns a dataframe of hits within a given mask around a given
        spatial location, sorted by distance from that location.
        """
        ds = raster.dataset.clone(df)
        xs, ys = (ds.dimension_values(kd) for kd in raster.kdims)
        dx, dy = (xs - x, ys - y)
        distances = pd.Series(dx * dx + dy * dy)
        return df.iloc[distances.argsort().values]