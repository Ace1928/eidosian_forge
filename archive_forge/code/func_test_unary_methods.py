import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_unary_methods(self):
    array = np.array([-1, 0, 1, 2])
    array_like = ArrayLike(array)
    for op in [operator.neg, operator.pos, abs, operator.invert]:
        _assert_equal_type_and_value(op(array_like), ArrayLike(op(array)))