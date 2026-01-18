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
def test_printfloat(self):
    output, _ = self.run_code(printfloat_usecase)
    expected_cases = ['0 23 34.750000 321', '0 23 34.75 321']
    self.assertIn(output.strip(), expected_cases)