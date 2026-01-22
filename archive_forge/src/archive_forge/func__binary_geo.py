import warnings
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import shapely.geos
import shapely.ops
import shapely.validation
import shapely.wkb
import shapely.wkt
from shapely.geometry.base import BaseGeometry
from . import _compat as compat
def _binary_geo(op, left, right):
    """Apply geometry-valued operation

    Supports:

    -   difference
    -   symmetric_difference
    -   intersection
    -   union

    Parameters
    ----------
    op: string
    right: np.array[geoms] or single shapely BaseGeoemtry
    """
    if isinstance(right, BaseGeometry):
        data = np.empty(len(left), dtype=object)
        with compat.ignore_shapely2_warnings():
            data[:] = [getattr(s, op)(right) if s is not None and right is not None else None for s in left]
        return data
    elif isinstance(right, np.ndarray):
        if len(left) != len(right):
            msg = 'Lengths of inputs do not match. Left: {0}, Right: {1}'.format(len(left), len(right))
            raise ValueError(msg)
        data = np.empty(len(left), dtype=object)
        with compat.ignore_shapely2_warnings():
            data[:] = [getattr(this_elem, op)(other_elem) if this_elem is not None and other_elem is not None else None for this_elem, other_elem in zip(left, right)]
        return data
    else:
        raise TypeError('Type not known: {0} vs {1}'.format(type(left), type(right)))