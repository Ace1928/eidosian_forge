import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
class SubArrayLike(ArrayLike):
    """Should take precedence over ArrayLike."""