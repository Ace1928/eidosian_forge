from __future__ import annotations
import re
import numpy as np
from tlz import concat, merge, unique
from dask.array.core import Array, apply_infer_dtype, asarray, blockwise, getitem
from dask.array.utils import meta_from_array
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
def _as_gufunc(pyfunc):
    return gufunc(pyfunc, signature=signature, **kwargs)