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
def _search_matching_signature(self, idtypes):
    """
        Given the input types in `idtypes`, return a compatible sequence of
        types that is defined in `kernelmap`.

        Note: Ordering is guaranteed by `kernelmap` being a OrderedDict
        """
    for sig in self.kernelmap.keys():
        if all((np.can_cast(actual, desired) for actual, desired in zip(sig, idtypes))):
            return sig
    else:
        raise TypeError('no matching signature')