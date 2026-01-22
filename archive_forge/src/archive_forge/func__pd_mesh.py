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
def _pd_mesh(vertices, simplices):
    """Helper for ``datashader.utils.mesh()``. Both arguments are assumed to be
    Pandas DataFrame objects.
    """
    winding = [0, 1, 2]
    first_tri = vertices.values[simplices.values[0, winding].astype(np.int64), :2]
    a, b, c = first_tri
    if np.cross(b - a, c - a).item() >= 0:
        winding = [0, 2, 1]
    vertex_idxs = simplices.values[:, winding]
    if not vertex_idxs.dtype == 'int64':
        vertex_idxs = vertex_idxs.astype(np.int64)
    vals = np.take(vertices.values, vertex_idxs, axis=0)
    vals = vals.reshape(np.prod(vals.shape[:2]), vals.shape[2])
    res = pd.DataFrame(vals, columns=vertices.columns)
    verts_have_weights = len(vertices.columns) > 2
    if not verts_have_weights:
        weight_col = simplices.columns[3]
        res[weight_col] = simplices.values[:, 3].repeat(3)
    return res