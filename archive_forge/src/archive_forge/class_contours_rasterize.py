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
class contours_rasterize(aggregate):
    """
    Rasterizes the Contours element by weighting the aggregation by
    the iso-contour levels if a value dimension is defined, otherwise
    default to any aggregator.
    """
    aggregator = param.ClassSelector(default=rd.mean(), class_=(rd.Reduction, rd.summary, str))

    @classmethod
    def _get_aggregator(cls, element, agg, add_field=True):
        if not element.vdims and agg.column is None and (not isinstance(agg, (rd.count, rd.any))):
            return ds.any()
        return super()._get_aggregator(element, agg, add_field)