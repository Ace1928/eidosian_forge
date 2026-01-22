import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _expit(X):
    xp, _ = get_namespace(X)
    if _is_numpy_namespace(xp):
        return xp.asarray(special.expit(numpy.asarray(X)))
    return 1.0 / (1.0 + xp.exp(-X))