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
def _gen_src_for_indexing(aref, adims, atype):
    return '{aref}[{sliced}]'.format(aref=aref, sliced=_gen_src_index(adims, atype))