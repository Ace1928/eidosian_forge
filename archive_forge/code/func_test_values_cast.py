import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
def test_values_cast(self):
    test1 = np.array([self.ucs_value * self.ulen] * 2, dtype='U%s' % self.ulen)
    test2 = np.repeat(test1, 2)[::2]
    for ua in (test1, test2):
        ua2 = ua.astype(dtype=ua.dtype.newbyteorder())
        assert_((ua == ua2).all())
        assert_(ua[-1] == ua2[-1])
        ua3 = ua2.astype(dtype=ua.dtype)
        assert_equal(ua, ua3)