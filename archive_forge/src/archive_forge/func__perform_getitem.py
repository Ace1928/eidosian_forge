import sys
import warnings
import numpy as np
import param
from .. import util
from ..element import Element
from ..ndmapping import NdMapping
from .util import finite_range
@classmethod
def _perform_getitem(cls, dataset, indices):
    ds = dataset
    indices = util.wrap_tuple(indices)
    if not ds.interface.gridded:
        raise IndexError('Cannot use ndloc on non nd-dimensional datastructure')
    selected = dataset.interface.ndloc(ds, indices)
    if np.isscalar(selected):
        return selected
    params = {}
    if hasattr(ds, 'bounds'):
        params['bounds'] = None
    return dataset.clone(selected, datatype=[ds.interface.datatype] + ds.datatype, **params)