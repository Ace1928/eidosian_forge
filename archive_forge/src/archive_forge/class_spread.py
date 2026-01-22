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
class spread(SpreadingOperation):
    """
    Spreading expands each pixel in an Image based Element a certain
    number of pixels on all sides according to a given shape, merging
    pixels using a specified compositing operator. This can be useful
    to make sparse plots more visible.

    See the datashader documentation for more detail:

    http://datashader.org/api.html#datashader.transfer_functions.spread
    """
    px = param.Integer(default=1, doc='\n        Number of pixels to spread on all sides.')

    def _apply_spreading(self, array):
        return tf.spread(array, px=self.p.px, how=self.p.how, shape=self.p.shape)