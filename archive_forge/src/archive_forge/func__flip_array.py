from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
def _flip_array(array, xflip, yflip):
    if yflip:
        array = array[..., ::-1, :]
    if xflip:
        array = array[..., :, ::-1]
    return array