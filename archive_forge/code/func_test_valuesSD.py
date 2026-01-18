import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal
def test_valuesSD(self):
    ua = np.array([self.ucs_value * self.ulen] * 2, dtype='U%s' % self.ulen)
    ua2 = ua.newbyteorder()
    assert_((ua != ua2).all())
    assert_(ua[-1] != ua2[-1])
    ua3 = ua2.newbyteorder()
    assert_equal(ua, ua3)