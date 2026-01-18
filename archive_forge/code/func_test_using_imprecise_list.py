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
def test_using_imprecise_list(self):
    """
        Type inference should report informative error about untyped list.
        TODO: #2931
        """
    with self.assertRaises(TypingError) as raises:
        njit(())(using_imprecise_list)
    errmsg = str(raises.exception)
    self.assertIn('Undecided type', errmsg)