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
def uint8_to_uint32(cls, img):
    shape = img.shape
    flat_shape = np.multiply.reduce(shape[:2])
    if shape[-1] == 3:
        img = np.dstack([img, np.ones(shape[:2], dtype='uint8') * 255])
    rgb = img.reshape((flat_shape, 4)).view('uint32').reshape(shape[:2])
    return rgb