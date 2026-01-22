from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def _get_actual_args(self):
    """Return the actual arguments
        Casts scalar arguments to np.array.
        """
    for i in self.scalarpos:
        self.arrays[i] = np.array([self.args[i]], dtype=self.argtypes[i])
    return self.arrays