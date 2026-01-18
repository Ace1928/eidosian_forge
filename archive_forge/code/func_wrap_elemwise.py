from __future__ import annotations
from functools import partial
from operator import getitem
import numpy as np
from dask import core
from dask.array.core import Array, apply_infer_dtype, asarray, blockwise, elemwise
from dask.base import is_dask_collection, normalize_token
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, funcname
def wrap_elemwise(numpy_ufunc, source=np):
    """Wrap up numpy function into dask.array"""

    def wrapped(*args, **kwargs):
        dsk = [arg for arg in args if hasattr(arg, '_elemwise')]
        if len(dsk) > 0:
            return dsk[0]._elemwise(numpy_ufunc, *args, **kwargs)
        else:
            return numpy_ufunc(*args, **kwargs)
    wrapped.__name__ = numpy_ufunc.__name__
    return derived_from(source)(wrapped)