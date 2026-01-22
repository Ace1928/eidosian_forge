import numpy as np
from numpy.testing import assert_equal
class TestDot:

    def test_matscalar(self):
        b1 = np.matrix(np.ones((3, 3), dtype=complex))
        assert_equal(b1 * 1.0, b1)