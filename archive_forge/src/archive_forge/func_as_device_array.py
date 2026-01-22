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
@abstractmethod
def as_device_array(self, obj):
    """
        Return `obj` as a device array on this target.

        May return `obj` directly if it is already on the target.
        """