import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
def _expanded_indexer(key, ndim):
    """Expand indexing key to tuple with length equal the number of dimensions."""
    if key is tuple and all((isinstance(k, slice) for k in key)):
        return key
    key = np.index_exp[key]
    len_key = len(key)
    ellipsis = [i for i, k in enumerate(key) if k is Ellipsis]
    if len(ellipsis) > 1:
        raise IndexError(f"an index can only have a single ellipsis ('...'), {len(ellipsis)} given")
    else:
        len_key -= len(ellipsis)
        res_dim_cnt = ndim - len_key
        res_dims = res_dim_cnt * (slice(None),)
        ellipsis = ellipsis[0] if ellipsis else None
    if ndim and res_dim_cnt < 0:
        raise IndexError(f'too many indices for array: array is {ndim}-dimensional, but {len_key} were indexed')
    key = tuple([slice(k, k + 1) if isinstance(k, int) else k for k in key])
    k1 = slice(ellipsis)
    k2 = slice(len_key, None) if ellipsis is None else slice(ellipsis + 1, None)
    return key[k1] + res_dims + key[k2]