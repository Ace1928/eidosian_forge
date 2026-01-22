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
class mr_class(MAxisConcatenator):
    """
    Translate slice objects to concatenation along the first axis.

    This is the masked array version of `lib.index_tricks.RClass`.

    See Also
    --------
    lib.index_tricks.RClass

    Examples
    --------
    >>> np.ma.mr_[np.ma.array([1,2,3]), 0, 0, np.ma.array([4,5,6])]
    masked_array(data=[1, 2, 3, ..., 4, 5, 6],
                 mask=False,
           fill_value=999999)

    """

    def __init__(self):
        MAxisConcatenator.__init__(self, 0)