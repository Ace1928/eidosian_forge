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
class _TimelikeFormat:

    def __init__(self, data):
        non_nat = data[~isnat(data)]
        if len(non_nat) > 0:
            max_str_len = max(len(self._format_non_nat(np.max(non_nat))), len(self._format_non_nat(np.min(non_nat))))
        else:
            max_str_len = 0
        if len(non_nat) < data.size:
            max_str_len = max(max_str_len, 5)
        self._format = '%{}s'.format(max_str_len)
        self._nat = "'NaT'".rjust(max_str_len)

    def _format_non_nat(self, x):
        raise NotImplementedError

    def __call__(self, x):
        if isnat(x):
            return self._nat
        else:
            return self._format % self._format_non_nat(x)