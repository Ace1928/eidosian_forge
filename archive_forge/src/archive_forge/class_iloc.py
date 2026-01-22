import sys
import warnings
import numpy as np
import param
from .. import util
from ..element import Element
from ..ndmapping import NdMapping
from .util import finite_range
class iloc(Accessor):
    """
    iloc is small wrapper object that allows row, column based
    indexing into a Dataset using the ``.iloc`` property.  It supports
    the usual numpy and pandas iloc indexing semantics including
    integer indices, slices, lists and arrays of values. For more
    information see the ``Dataset.iloc`` property docstring.
    """

    @classmethod
    def _perform_getitem(cls, dataset, index):
        index = util.wrap_tuple(index)
        if len(index) == 1:
            index = (index[0], slice(None))
        elif len(index) > 2:
            raise IndexError('Tabular index not understood, index must be at most length 2.')
        rows, cols = index
        if rows is Ellipsis:
            rows = slice(None)
        data = dataset.interface.iloc(dataset, (rows, cols))
        kdims = dataset.kdims
        vdims = dataset.vdims
        if util.isscalar(data):
            return data
        elif cols == slice(None):
            pass
        else:
            if isinstance(cols, slice):
                dims = dataset.dimensions()[index[1]]
            elif np.isscalar(cols):
                dims = [dataset.get_dimension(cols)]
            else:
                dims = [dataset.get_dimension(d) for d in cols]
            kdims = [d for d in dims if d in kdims]
            vdims = [d for d in dims if d in vdims]
        datatypes = util.unique_iterator([dataset.interface.datatype] + dataset.datatype)
        datatype = [dt for dt in datatypes if dt in Interface.interfaces and (not Interface.interfaces[dt].gridded)]
        if not datatype:
            datatype = ['dataframe', 'dictionary']
        return dataset.clone(data, kdims=kdims, vdims=vdims, datatype=datatype)