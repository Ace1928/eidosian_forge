import numpy as np
from numpy.testing import assert_array_equal, assert_
from scipy.io.matlab._mio_utils import squeeze_element, chars_to_strings
def test_squeeze_element():
    a = np.zeros((1, 3))
    assert_array_equal(np.squeeze(a), squeeze_element(a))
    sq_int = squeeze_element(np.zeros((1, 1), dtype=float))
    assert_(isinstance(sq_int, float))
    sq_sa = squeeze_element(np.zeros((1, 1), dtype=[('f1', 'f')]))
    assert_(isinstance(sq_sa, np.ndarray))
    sq_empty = squeeze_element(np.empty(0, np.uint8))
    assert sq_empty.dtype == np.uint8