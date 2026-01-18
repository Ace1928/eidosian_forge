from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import subprocess
import sys
import unittest
from numba import cuda
from numba import cuda
from numba import cuda
from numba import cuda
from numba import cuda
import numpy as np
@skip_on_cudasim('cudasim can print unlimited output')
def test_too_many_args(self):
    output, errors = self.run_code(print_too_many_usecase)
    expected_fmt_string = ' '.join(['%lld' for _ in range(33)])
    self.assertIn(expected_fmt_string, output)
    warn_msg = 'CUDA print() cannot print more than 32 items. The raw format string will be emitted by the kernel instead.'
    self.assertIn(warn_msg, errors)