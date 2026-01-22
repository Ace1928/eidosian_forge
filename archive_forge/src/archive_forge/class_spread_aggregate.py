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
class spread_aggregate(area_aggregate):
    """
    Aggregates Spread elements by filling the area between the lower
    and upper error band.
    """

    def _process(self, element, key=None):
        x, y = element.dimensions()[:2]
        df = PandasInterface.as_dframe(element)
        if df is element.data:
            df = df.copy()
        pos, neg = element.vdims[1:3] if len(element.vdims) > 2 else element.vdims[1:2] * 2
        yvals = df[y.name]
        df[y.name] = yvals + df[pos.name]
        df['_lower'] = yvals - df[neg.name]
        area = element.clone(df, vdims=[y, '_lower'] + element.vdims[3:], new_type=Area)
        return super()._process(area, key=None)