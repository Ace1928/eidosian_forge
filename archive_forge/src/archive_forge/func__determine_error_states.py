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
def _determine_error_states():
    errobj = geterrobj()
    bufsize = errobj[0]
    with errstate(invalid='call', over='ignore', divide='ignore', under='ignore'):
        invalid_call_errmask = geterrobj()[1]
    return [bufsize, invalid_call_errmask, None]