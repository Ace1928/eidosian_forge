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
def _init_bilinear(self, _, arr):
    weight = np.zeros(np.prod(arr.shape), dtype='float32')
    shape = arr.shape
    f = np.ceil(shape[3] / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(np.prod(shape)):
        x = i % shape[3]
        y = i // shape[3] % shape[2]
        weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    arr[:] = weight.reshape(shape)