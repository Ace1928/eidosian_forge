from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
class TestPointerArray:

    def test_1d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_1d.sav'), verbose=False)
        assert_equal(s.array1d.shape, (123,))
        assert_(np.all(s.array1d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array1d) == id(s.array1d[0])))

    def test_2d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_2d.sav'), verbose=False)
        assert_equal(s.array2d.shape, (22, 12))
        assert_(np.all(s.array2d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array2d) == id(s.array2d[0, 0])))

    def test_3d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_3d.sav'), verbose=False)
        assert_equal(s.array3d.shape, (11, 22, 12))
        assert_(np.all(s.array3d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array3d) == id(s.array3d[0, 0, 0])))

    def test_4d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_4d.sav'), verbose=False)
        assert_equal(s.array4d.shape, (4, 5, 8, 7))
        assert_(np.all(s.array4d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array4d) == id(s.array4d[0, 0, 0, 0])))

    def test_5d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_5d.sav'), verbose=False)
        assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
        assert_(np.all(s.array5d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array5d) == id(s.array5d[0, 0, 0, 0, 0])))

    def test_6d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_6d.sav'), verbose=False)
        assert_equal(s.array6d.shape, (3, 6, 4, 5, 3, 4))
        assert_(np.all(s.array6d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array6d) == id(s.array6d[0, 0, 0, 0, 0, 0])))

    def test_7d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_7d.sav'), verbose=False)
        assert_equal(s.array7d.shape, (2, 1, 2, 3, 4, 3, 2))
        assert_(np.all(s.array7d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array7d) == id(s.array7d[0, 0, 0, 0, 0, 0, 0])))

    def test_8d(self):
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_8d.sav'), verbose=False)
        assert_equal(s.array8d.shape, (4, 3, 2, 1, 2, 3, 5, 4))
        assert_(np.all(s.array8d == np.float32(4.0)))
        assert_(np.all(vect_id(s.array8d) == id(s.array8d[0, 0, 0, 0, 0, 0, 0, 0])))