from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
class TestPointerStructures:

    def test_scalars(self):
        s = readsav(path.join(DATA_PATH, 'struct_pointers.sav'), verbose=False)
        assert_identical(s.pointers.g, np.array(np.float32(4.0), dtype=np.object_))
        assert_identical(s.pointers.h, np.array(np.float32(4.0), dtype=np.object_))
        assert_(id(s.pointers.g[0]) == id(s.pointers.h[0]))

    def test_pointers_replicated(self):
        s = readsav(path.join(DATA_PATH, 'struct_pointers_replicated.sav'), verbose=False)
        assert_identical(s.pointers_rep.g, np.repeat(np.float32(4.0), 5).astype(np.object_))
        assert_identical(s.pointers_rep.h, np.repeat(np.float32(4.0), 5).astype(np.object_))
        assert_(np.all(vect_id(s.pointers_rep.g) == vect_id(s.pointers_rep.h)))

    def test_pointers_replicated_3d(self):
        s = readsav(path.join(DATA_PATH, 'struct_pointers_replicated_3d.sav'), verbose=False)
        s_expect = np.repeat(np.float32(4.0), 24).reshape(4, 3, 2).astype(np.object_)
        assert_identical(s.pointers_rep.g, s_expect)
        assert_identical(s.pointers_rep.h, s_expect)
        assert_(np.all(vect_id(s.pointers_rep.g) == vect_id(s.pointers_rep.h)))

    def test_arrays(self):
        s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays.sav'), verbose=False)
        assert_array_identical(s.arrays.g[0], np.repeat(np.float32(4.0), 2).astype(np.object_))
        assert_array_identical(s.arrays.h[0], np.repeat(np.float32(4.0), 3).astype(np.object_))
        assert_(np.all(vect_id(s.arrays.g[0]) == id(s.arrays.g[0][0])))
        assert_(np.all(vect_id(s.arrays.h[0]) == id(s.arrays.h[0][0])))
        assert_(id(s.arrays.g[0][0]) == id(s.arrays.h[0][0]))

    def test_arrays_replicated(self):
        s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays_replicated.sav'), verbose=False)
        assert_(s.arrays_rep.g.dtype.type is np.object_)
        assert_(s.arrays_rep.h.dtype.type is np.object_)
        assert_equal(s.arrays_rep.g.shape, (5,))
        assert_equal(s.arrays_rep.h.shape, (5,))
        for i in range(5):
            assert_array_identical(s.arrays_rep.g[i], np.repeat(np.float32(4.0), 2).astype(np.object_))
            assert_array_identical(s.arrays_rep.h[i], np.repeat(np.float32(4.0), 3).astype(np.object_))
            assert_(np.all(vect_id(s.arrays_rep.g[i]) == id(s.arrays_rep.g[0][0])))
            assert_(np.all(vect_id(s.arrays_rep.h[i]) == id(s.arrays_rep.h[0][0])))

    def test_arrays_replicated_3d(self):
        pth = path.join(DATA_PATH, 'struct_pointer_arrays_replicated_3d.sav')
        s = readsav(pth, verbose=False)
        assert_(s.arrays_rep.g.dtype.type is np.object_)
        assert_(s.arrays_rep.h.dtype.type is np.object_)
        assert_equal(s.arrays_rep.g.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.h.shape, (4, 3, 2))
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    assert_array_identical(s.arrays_rep.g[i, j, k], np.repeat(np.float32(4.0), 2).astype(np.object_))
                    assert_array_identical(s.arrays_rep.h[i, j, k], np.repeat(np.float32(4.0), 3).astype(np.object_))
                    g0 = vect_id(s.arrays_rep.g[i, j, k])
                    g1 = id(s.arrays_rep.g[0, 0, 0][0])
                    assert np.all(g0 == g1)
                    h0 = vect_id(s.arrays_rep.h[i, j, k])
                    h1 = id(s.arrays_rep.h[0, 0, 0][0])
                    assert np.all(h0 == h1)