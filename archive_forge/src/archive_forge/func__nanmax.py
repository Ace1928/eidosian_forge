import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _nanmax(X, axis=None):
    xp, _ = get_namespace(X)
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nanmax(X, axis=axis))
    else:
        mask = xp.isnan(X)
        X = xp.max(xp.where(mask, xp.asarray(-xp.inf, device=device(X)), X), axis=axis)
        mask = xp.all(mask, axis=axis)
        if xp.any(mask):
            X = xp.where(mask, xp.asarray(xp.nan), X)
        return X