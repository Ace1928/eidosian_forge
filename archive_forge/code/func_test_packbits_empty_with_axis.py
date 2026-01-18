import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain
def test_packbits_empty_with_axis():
    shapes = [((0,), [(0,)]), ((10, 20, 0), [(2, 20, 0), (10, 3, 0), (10, 20, 0)]), ((10, 0, 20), [(2, 0, 20), (10, 0, 20), (10, 0, 3)]), ((0, 10, 20), [(0, 10, 20), (0, 2, 20), (0, 10, 3)]), ((20, 0, 0), [(3, 0, 0), (20, 0, 0), (20, 0, 0)]), ((0, 20, 0), [(0, 20, 0), (0, 3, 0), (0, 20, 0)]), ((0, 0, 20), [(0, 0, 20), (0, 0, 20), (0, 0, 3)]), ((0, 0, 0), [(0, 0, 0), (0, 0, 0), (0, 0, 0)])]
    for dt in '?bBhHiIlLqQ':
        for in_shape, out_shapes in shapes:
            for ax, out_shape in enumerate(out_shapes):
                a = np.empty(in_shape, dtype=dt)
                b = np.packbits(a, axis=ax)
                assert_equal(b.dtype, np.uint8)
                assert_equal(b.shape, out_shape)