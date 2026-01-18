import numpy as np
from numba import from_dtype, njit, void
from numba.tests.support import TestCase
def test_record_misaligned(self):
    rec_dtype = np.dtype([('a', 'int32'), ('b', 'float64')])
    rec = from_dtype(rec_dtype)

    @njit((rec[:],))
    def foo(a):
        for i in range(a.size):
            a[i].a = a[i].b