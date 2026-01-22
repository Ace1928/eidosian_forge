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
def _gen_src_index(adims, atype):
    if adims > 0:
        return ','.join(['__tid__'] + [':'] * adims)
    elif isinstance(atype, types.Array) and atype.ndim - 1 == adims:
        return '__tid__:(__tid__ + 1)'
    else:
        return '__tid__'