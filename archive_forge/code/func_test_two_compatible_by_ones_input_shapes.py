import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_two_compatible_by_ones_input_shapes():
    data = [[[(1,), (3,)], (3,)], [[(1, 3), (3, 3)], (3, 3)], [[(3, 1), (3, 3)], (3, 3)], [[(1, 3), (3, 1)], (3, 3)], [[(1, 1), (3, 3)], (3, 3)], [[(1, 1), (1, 3)], (1, 3)], [[(1, 1), (3, 1)], (3, 1)], [[(1, 0), (0, 0)], (0, 0)], [[(0, 1), (0, 0)], (0, 0)], [[(1, 0), (0, 1)], (0, 0)], [[(1, 1), (0, 0)], (0, 0)], [[(1, 1), (1, 0)], (1, 0)], [[(1, 1), (0, 1)], (0, 1)]]
    for input_shapes, expected_shape in data:
        assert_shapes_correct(input_shapes, expected_shape)
        assert_shapes_correct(input_shapes[::-1], expected_shape)