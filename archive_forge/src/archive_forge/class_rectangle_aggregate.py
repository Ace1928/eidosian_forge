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
class rectangle_aggregate(geom_aggregate):
    """
    Aggregates Rectangle elements.
    """

    def _aggregate(self, cvs, df, x0, y0, x1, y1, agg_fn):
        return cvs.area(df, x=[x0, x1], y=y0, y_stack=y1, agg=agg_fn, axis=1)