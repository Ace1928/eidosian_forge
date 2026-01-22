import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
class MAxisConcatenator(AxisConcatenator):
    """
    Translate slice objects to concatenation along an axis.

    For documentation on usage, see `mr_class`.

    See Also
    --------
    mr_class

    """
    concatenate = staticmethod(concatenate)

    @classmethod
    def makemat(cls, arr):
        data = super().makemat(arr.data, copy=False)
        return array(data, mask=arr.mask)

    def __getitem__(self, key):
        if isinstance(key, str):
            raise MAError('Unavailable for masked array.')
        return super().__getitem__(key)