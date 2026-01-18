import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_two_global_rec_arrs_npm(self):
    self.check_two_global_rec_arrs(nopython=True)