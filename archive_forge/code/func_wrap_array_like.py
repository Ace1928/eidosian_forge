import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def wrap_array_like(result):
    if type(result) is tuple:
        return tuple((ArrayLike(r) for r in result))
    else:
        return ArrayLike(result)