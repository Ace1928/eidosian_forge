import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
def test_unknown_module(self):
    with self.assertRaises(TypingError) as raises:
        njit(())(unknown_module)
    self.assertIn("name 'numpyz' is not defined", str(raises.exception))