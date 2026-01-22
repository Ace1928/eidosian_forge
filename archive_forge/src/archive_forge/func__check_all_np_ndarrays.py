import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _check_all_np_ndarrays(out):
    """Check if ndarrays/symbols in out are all np.ndarray/np._Symbol."""
    from ..numpy import ndarray as np_ndarray
    from ..symbol.numpy import _Symbol as np_symbol
    from ..symbol import Symbol as nd_symbol
    from ..ndarray import NDArray as nd_ndarray
    if isinstance(out, (nd_ndarray, nd_symbol)) and (not isinstance(out, (np_ndarray, np_symbol))):
        raise TypeError("Block's output ndarrays/symbols must be of type `mxnet.numpy.ndarray` or `mxnet.symbol.numpy._Symbol`, while got output type {}".format(str(type(out))))
    elif isinstance(out, (list, tuple)):
        for i in out:
            _check_all_np_ndarrays(i)