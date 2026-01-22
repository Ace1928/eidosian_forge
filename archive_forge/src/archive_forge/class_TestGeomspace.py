import pytest
from numpy import (
from numpy.testing import (
class TestGeomspace:

    def test_basic(self):
        y = geomspace(1, 1000000.0)
        assert_(len(y) == 50)
        y = geomspace(1, 1000000.0, num=100)
        assert_(y[-1] == 10 ** 6)
        y = geomspace(1, 1000000.0, endpoint=False)
        assert_(y[-1] < 10 ** 6)
        y = geomspace(1, 1000000.0, num=7)
        assert_array_equal(y, [1, 10, 100, 1000.0, 10000.0, 100000.0, 1000000.0])
        y = geomspace(8, 2, num=3)
        assert_allclose(y, [8, 4, 2])
        assert_array_equal(y.imag, 0)
        y = geomspace(-1, -100, num=3)
        assert_array_equal(y, [-1, -10, -100])
        assert_array_equal(y.imag, 0)
        y = geomspace(-100, -1, num=3)
        assert_array_equal(y, [-100, -10, -1])
        assert_array_equal(y.imag, 0)

    def test_boundaries_match_start_and_stop_exactly(self):
        start = 0.3
        stop = 20.3
        y = geomspace(start, stop, num=1)
        assert_equal(y[0], start)
        y = geomspace(start, stop, num=1, endpoint=False)
        assert_equal(y[0], start)
        y = geomspace(start, stop, num=3)
        assert_equal(y[0], start)
        assert_equal(y[-1], stop)
        y = geomspace(start, stop, num=3, endpoint=False)
        assert_equal(y[0], start)

    def test_nan_interior(self):
        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4)
        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:-1]).all())
        assert_equal(y[3], 3.0)
        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4, endpoint=False)
        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:]).all())

    def test_complex(self):
        y = geomspace(1j, 16j, num=5)
        assert_allclose(y, [1j, 2j, 4j, 8j, 16j])
        assert_array_equal(y.real, 0)
        y = geomspace(-4j, -324j, num=5)
        assert_allclose(y, [-4j, -12j, -36j, -108j, -324j])
        assert_array_equal(y.real, 0)
        y = geomspace(1 + 1j, 1000 + 1000j, num=4)
        assert_allclose(y, [1 + 1j, 10 + 10j, 100 + 100j, 1000 + 1000j])
        y = geomspace(-1 + 1j, -1000 + 1000j, num=4)
        assert_allclose(y, [-1 + 1j, -10 + 10j, -100 + 100j, -1000 + 1000j])
        y = geomspace(-1, 1, num=3, dtype=complex)
        assert_allclose(y, [-1, 1j, +1])
        y = geomspace(0 + 3j, -3 + 0j, 3)
        assert_allclose(y, [0 + 3j, -3 / sqrt(2) + 3j / sqrt(2), -3 + 0j])
        y = geomspace(0 + 3j, 3 + 0j, 3)
        assert_allclose(y, [0 + 3j, 3 / sqrt(2) + 3j / sqrt(2), 3 + 0j])
        y = geomspace(-3 + 0j, 0 - 3j, 3)
        assert_allclose(y, [-3 + 0j, -3 / sqrt(2) - 3j / sqrt(2), 0 - 3j])
        y = geomspace(0 + 3j, -3 + 0j, 3)
        assert_allclose(y, [0 + 3j, -3 / sqrt(2) + 3j / sqrt(2), -3 + 0j])
        y = geomspace(-2 - 3j, 5 + 7j, 7)
        assert_allclose(y, [-2 - 3j, -0.29058977 - 4.15771027j, 2.08885354 - 4.34146838j, 4.58345529 - 3.16355218j, 6.41401745 - 0.55233457j, 6.75707386 + 3.11795092j, 5 + 7j])
        y = geomspace(3j, -5, 2)
        assert_allclose(y, [3j, -5])
        y = geomspace(-5, 3j, 2)
        assert_allclose(y, [-5, 3j])

    def test_dtype(self):
        y = geomspace(1, 1000000.0, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = geomspace(1, 1000000.0, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = geomspace(1, 1000000.0, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))
        y = geomspace(1, 1000000.0, dtype=float)
        assert_equal(y.dtype, dtype('float_'))
        y = geomspace(1, 1000000.0, dtype=complex)
        assert_equal(y.dtype, dtype('complex'))

    def test_start_stop_array_scalar(self):
        lim1 = array([120, 100], dtype='int8')
        lim2 = array([-120, -100], dtype='int8')
        lim3 = array([1200, 1000], dtype='uint16')
        t1 = geomspace(lim1[0], lim1[1], 5)
        t2 = geomspace(lim2[0], lim2[1], 5)
        t3 = geomspace(lim3[0], lim3[1], 5)
        t4 = geomspace(120.0, 100.0, 5)
        t5 = geomspace(-120.0, -100.0, 5)
        t6 = geomspace(1200.0, 1000.0, 5)
        assert_allclose(t1, t4, rtol=0.01)
        assert_allclose(t2, t5, rtol=0.01)
        assert_allclose(t3, t6, rtol=1e-05)

    def test_start_stop_array(self):
        start = array([1.0, 32.0, 1j, -4j, 1 + 1j, -1])
        stop = array([10000.0, 2.0, 16j, -324j, 10000 + 10000j, 1])
        t1 = geomspace(start, stop, 5)
        t2 = stack([geomspace(_start, _stop, 5) for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = geomspace(start, stop[0], 5)
        t4 = stack([geomspace(_start, stop[0], 5) for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = geomspace(start, stop, 5, axis=-1)
        assert_equal(t5, t2.T)

    def test_physical_quantities(self):
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        assert_equal(geomspace(a, b), geomspace(1.0, 5.0))

    def test_subclass(self):
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        gs = geomspace(a, b)
        assert type(gs) is PhysicalQuantity2
        assert_equal(gs, geomspace(1.0, 7.0))
        gs = geomspace(a, b, 1)
        assert type(gs) is PhysicalQuantity2
        assert_equal(gs, geomspace(1.0, 7.0, 1))

    def test_bounds(self):
        assert_raises(ValueError, geomspace, 0, 10)
        assert_raises(ValueError, geomspace, 10, 0)
        assert_raises(ValueError, geomspace, 0, 0)