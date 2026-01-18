from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_scalars_replicated_3d(self):
    s = readsav(path.join(DATA_PATH, 'struct_scalars_replicated_3d.sav'), verbose=False)
    assert_identical(s.scalars_rep.a, np.repeat(np.int16(1), 24).reshape(4, 3, 2))
    assert_identical(s.scalars_rep.b, np.repeat(np.int32(2), 24).reshape(4, 3, 2))
    assert_identical(s.scalars_rep.c, np.repeat(np.float32(3.0), 24).reshape(4, 3, 2))
    assert_identical(s.scalars_rep.d, np.repeat(np.float64(4.0), 24).reshape(4, 3, 2))
    assert_identical(s.scalars_rep.e, np.repeat(b'spam', 24).reshape(4, 3, 2).astype(object))
    assert_identical(s.scalars_rep.f, np.repeat(np.complex64(-1.0 + 3j), 24).reshape(4, 3, 2))