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
def _init_loc_bias(self, _, arr):
    shape = arr.shape
    assert shape[0] == 6
    arr[:] = np.array([1.0, 0, 0, 0, 1.0, 0])