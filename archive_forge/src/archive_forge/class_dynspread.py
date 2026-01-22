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
class dynspread(SpreadingOperation):
    """
    Spreading expands each pixel in an Image based Element a certain
    number of pixels on all sides according to a given shape, merging
    pixels using a specified compositing operator. This can be useful
    to make sparse plots more visible. Dynamic spreading determines
    how many pixels to spread based on a density heuristic.

    See the datashader documentation for more detail:

    http://datashader.org/api.html#datashader.transfer_functions.dynspread
    """
    max_px = param.Integer(default=3, doc='\n        Maximum number of pixels to spread on all sides.')
    threshold = param.Number(default=0.5, bounds=(0, 1), doc='\n        When spreading, determines how far to spread.\n        Spreading starts at 1 pixel, and stops when the fraction\n        of adjacent non-empty pixels reaches this threshold.\n        Higher values give more spreading, up to the max_px\n        allowed.')

    def _apply_spreading(self, array):
        return tf.dynspread(array, max_px=self.p.max_px, threshold=self.p.threshold, how=self.p.how, shape=self.p.shape)