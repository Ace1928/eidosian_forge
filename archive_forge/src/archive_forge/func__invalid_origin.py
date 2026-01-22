from collections.abc import Iterable
import numbers
import warnings
import numpy
import operator
from scipy._lib._util import normalize_axis_index
from . import _ni_support
from . import _nd_image
from . import _ni_docstrings
def _invalid_origin(origin, lenw):
    return origin < -(lenw // 2) or origin > (lenw - 1) // 2