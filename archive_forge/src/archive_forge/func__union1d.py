import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _union1d(a, b, xp):
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.union1d(a, b))
    assert a.ndim == b.ndim == 1
    return xp.unique_values(xp.concat([xp.unique_values(a), xp.unique_values(b)]))