import numpy as np
from numba import cuda
from numba.cuda.args import wrap_arg
from numba.cuda.testing import CUDATestCase
import unittest
def test_record_inout(self):
    host_rec = np.zeros(1, dtype=recordtype)
    self.set_record_to_three[1, 1](cuda.InOut(host_rec))
    self.assertEqual(3, host_rec[0]['b'])