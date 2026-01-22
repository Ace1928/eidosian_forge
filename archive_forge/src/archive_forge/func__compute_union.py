import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
def _compute_union(w1, w2):
    """Compute the union of two windows"""
    col_off = min(w1.col_off, w2.col_off)
    row_off = min(w1.row_off, w2.row_off)
    width = max(w1.col_off + w1.width, w2.col_off + w2.width) - col_off
    height = max(w1.row_off + w1.height, w2.row_off + w2.height) - row_off
    return (col_off, row_off, width, height)