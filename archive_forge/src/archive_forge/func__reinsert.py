from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _reinsert(indices, arr, new):
    trail_dim = arr.shape[-1] + len(indices)
    new_arr = np.empty(arr.shape[:-1] + (trail_dim,))
    idx_arr, idx_insert = (0, 0)
    for glbl in range(trail_dim):
        if glbl in indices:
            new_arr[..., glbl] = new[..., idx_insert]
            idx_insert += 1
        else:
            new_arr[..., glbl] = arr[..., idx_arr]
            idx_arr += 1
    return new_arr