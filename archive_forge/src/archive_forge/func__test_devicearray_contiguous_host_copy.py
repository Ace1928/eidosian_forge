import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def _test_devicearray_contiguous_host_copy(self, a_c, a_f):
    """
        Checks host->device memcpys
        """
    self.assertTrue(a_c.flags.c_contiguous)
    self.assertTrue(a_f.flags.f_contiguous)
    for original, copy in [(a_f, a_f), (a_f, a_c), (a_c, a_f), (a_c, a_c)]:
        msg = '%s => %s' % ('C' if original.flags.c_contiguous else 'F', 'C' if copy.flags.c_contiguous else 'F')
        d = cuda.to_device(original)
        d.copy_to_device(copy)
        self.assertTrue(np.all(d.copy_to_host() == a_c), msg=msg)
        self.assertTrue(np.all(d.copy_to_host() == a_f), msg=msg)