import functools
import operator
import warnings
from typing import NamedTuple, Any
from .._utils import set_module
from numpy.core import (
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
from numpy.lib.twodim_base import triu, eye
from numpy.linalg import _umath_linalg
from numpy._typing import NDArray
def _assert_finite(*arrays):
    for a in arrays:
        if not isfinite(a).all():
            raise LinAlgError('Array must not contain infs or NaNs')