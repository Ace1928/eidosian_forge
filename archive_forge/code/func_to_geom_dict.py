import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
def to_geom_dict(eltype, data, kdims, vdims, interface=None):
    """Converts data from any list format to a dictionary based format.

    Args:
        eltype: Element type to convert
        data: The original data
        kdims: The declared key dimensions
        vdims: The declared value dimensions

    Returns:
        A list of dictionaries containing geometry coordinates and values.
    """
    from . import Dataset
    xname, yname = (kd.name for kd in kdims[:2])
    if isinstance(data, dict):
        data = {k: v if isscalar(v) else _asarray(v) for k, v in data.items()}
        return data
    new_el = Dataset(data, kdims, vdims)
    if new_el.interface is interface:
        return new_el.data
    new_dict = {}
    for d in new_el.dimensions():
        if d in (xname, yname):
            scalar = False
        else:
            scalar = new_el.interface.isscalar(new_el, d)
        vals = new_el.dimension_values(d, not scalar)
        new_dict[d.name] = vals[0] if scalar else vals
    return new_dict