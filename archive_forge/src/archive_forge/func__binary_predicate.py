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
def _binary_predicate(op, left, right, *args, **kwargs):
    """Binary operation on np.array[geoms] that returns a boolean ndarray

    Supports:

    -  contains
    -  disjoint
    -  intersects
    -  touches
    -  crosses
    -  within
    -  overlaps
    -  covers
    -  covered_by
    -  equals

    Parameters
    ----------
    op: string
    right: np.array[geoms] or single shapely BaseGeoemtry
    """
    if isinstance(right, BaseGeometry):
        data = [getattr(s, op)(right, *args, **kwargs) if s is not None else False for s in left]
        return np.array(data, dtype=bool)
    elif isinstance(right, np.ndarray):
        data = [getattr(this_elem, op)(other_elem, *args, **kwargs) if not (this_elem is None or other_elem is None) else False for this_elem, other_elem in zip(left, right)]
        return np.array(data, dtype=bool)
    else:
        raise TypeError('Type not known: {0} vs {1}'.format(type(left), type(right)))