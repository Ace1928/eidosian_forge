import re
import logging
import warnings
import json
from math import sqrt
import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
from . import registry
from . import ndarray
from . util import is_np_array
from . import numpy as _mx_np  # pylint: disable=reimported
def _legacy_init(self, name, arr):
    """Legacy initialization method.

        Parameters
        ----------
        name : str
            Name of corresponding NDArray.

        arr : NDArray
            NDArray to be initialized.
        """
    warnings.warn('\x1b[91mCalling initializer with init(str, NDArray) has been deprecated.please use init(mx.init.InitDesc(...), NDArray) instead.\x1b[0m', DeprecationWarning, stacklevel=3)
    if not isinstance(name, string_types):
        raise TypeError('name must be string')
    if not isinstance(arr, NDArray):
        raise TypeError('arr must be NDArray')
    if name.startswith('upsampling'):
        self._init_bilinear(name, arr)
    elif name.startswith('stn_loc') and name.endswith('weight'):
        self._init_zero(name, arr)
    elif name.startswith('stn_loc') and name.endswith('bias'):
        self._init_loc_bias(name, arr)
    elif name.endswith('bias'):
        self._init_bias(name, arr)
    elif name.endswith('gamma'):
        self._init_gamma(name, arr)
    elif name.endswith('beta'):
        self._init_beta(name, arr)
    elif name.endswith('weight'):
        self._init_weight(name, arr)
    elif name.endswith('moving_mean'):
        self._init_zero(name, arr)
    elif name.endswith('moving_var'):
        self._init_one(name, arr)
    elif name.endswith('moving_inv_var'):
        self._init_zero(name, arr)
    elif name.endswith('moving_avg'):
        self._init_zero(name, arr)
    elif name.endswith('min'):
        self._init_zero(name, arr)
    elif name.endswith('max'):
        self._init_one(name, arr)
    else:
        self._init_default(name, arr)