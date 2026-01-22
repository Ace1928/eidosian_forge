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
def _fill_argtypes(self):
    """
        Get dtypes
        """
    for i, ary in enumerate(self.arrays):
        if ary is not None:
            dtype = getattr(ary, 'dtype')
            if dtype is None:
                dtype = np.asarray(ary).dtype
            self.argtypes[i] = dtype