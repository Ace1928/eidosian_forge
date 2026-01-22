import operator
import functools
import warnings
import numpy as np
from numpy.core.multiarray import dragon4_positional, dragon4_scientific
from numpy.core.umath import absolute
class RankWarning(UserWarning):
    """Issued by chebfit when the design matrix is rank deficient."""
    pass