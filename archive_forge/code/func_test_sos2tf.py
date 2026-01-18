import numpy as np
from numpy.testing import assert_
from scipy.signal import (decimate,
def test_sos2tf():
    sos_f32 = np.array([[4, 5, 6, 1, 2, 3]], dtype=np.float32)
    b, a = sos2tf(sos_f32)
    assert_(b.dtype == np.float32)
    assert_(a.dtype == np.float32)