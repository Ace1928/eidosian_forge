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
@register
class LSTMBias(Initializer):
    """Initialize all biases of an LSTMCell to 0.0 except for
    the forget gate whose bias is set to custom value.

    Parameters
    ----------
    forget_bias: float, default 1.0
        bias for the forget gate. Jozefowicz et al. 2015 recommends
        setting this to 1.0.
    """

    def __init__(self, forget_bias=1.0):
        super(LSTMBias, self).__init__(forget_bias=forget_bias)
        self.forget_bias = forget_bias

    def _init_weight(self, name, arr):
        arr[:] = 0.0
        num_hidden = int(arr.shape[0] / 4)
        arr[num_hidden:2 * num_hidden] = self.forget_bias