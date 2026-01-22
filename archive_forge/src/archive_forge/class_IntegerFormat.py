import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
class IntegerFormat:

    def __init__(self, data):
        if data.size > 0:
            max_str_len = max(len(str(np.max(data))), len(str(np.min(data))))
        else:
            max_str_len = 0
        self.format = '%{}d'.format(max_str_len)

    def __call__(self, x):
        return self.format % x