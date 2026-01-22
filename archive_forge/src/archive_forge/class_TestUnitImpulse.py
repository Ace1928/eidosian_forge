import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
class TestUnitImpulse:

    def test_no_index(self):
        assert_array_equal(waveforms.unit_impulse(7), [1, 0, 0, 0, 0, 0, 0])
        assert_array_equal(waveforms.unit_impulse((3, 3)), [[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_index(self):
        assert_array_equal(waveforms.unit_impulse(10, 3), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert_array_equal(waveforms.unit_impulse((3, 3), (1, 1)), [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        imp = waveforms.unit_impulse((4, 4), 2)
        assert_array_equal(imp, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]))

    def test_mid(self):
        assert_array_equal(waveforms.unit_impulse((3, 3), 'mid'), [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_array_equal(waveforms.unit_impulse(9, 'mid'), [0, 0, 0, 0, 1, 0, 0, 0, 0])

    def test_dtype(self):
        imp = waveforms.unit_impulse(7)
        assert_(np.issubdtype(imp.dtype, np.floating))
        imp = waveforms.unit_impulse(5, 3, dtype=int)
        assert_(np.issubdtype(imp.dtype, np.integer))
        imp = waveforms.unit_impulse((5, 2), (3, 1), dtype=complex)
        assert_(np.issubdtype(imp.dtype, np.complexfloating))