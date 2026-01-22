import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def _ensure_objmode(self, disp):
    self.assertTrue(disp.signatures)
    self.assertFalse(disp.nopython_signatures)
    return disp