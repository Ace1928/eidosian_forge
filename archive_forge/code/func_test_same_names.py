import multiprocessing
import os
import shutil
import unittest
import warnings
from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.tests.support import SerialMixin
from numba.tests.test_caching import (DispatcherCacheUsecasesTest,
def test_same_names(self):
    mod = self.import_module()
    f = mod.renamed_function1
    self.assertPreciseEqual(f(2), 4)
    f = mod.renamed_function2
    self.assertPreciseEqual(f(2), 8)