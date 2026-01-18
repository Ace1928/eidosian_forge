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
def test_return_type_unification(self):
    with self.assertRaises(TypingError) as raises:
        njit((types.int32,))(impossible_return_type)
    msg = "Can't unify return type from the following types: Tuple(), complex128"
    self.assertIn(msg, str(raises.exception))