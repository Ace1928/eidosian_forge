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
def _empty_df(cls, dataset):
    if 'dask' in dataset.interface.datatype:
        return dataset.data._meta.iloc[:0]
    elif dataset.interface.datatype in ['pandas', 'geopandas', 'spatialpandas']:
        return dataset.data.head(0)
    return dataset.iloc[:0].dframe()