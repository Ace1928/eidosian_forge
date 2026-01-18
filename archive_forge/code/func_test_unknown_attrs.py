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
def test_unknown_attrs(self):
    try:
        njit((types.int32,))(bar)
    except TypingError as e:
        self.assertIn("Unknown attribute 'a' of type int32", str(e))
    else:
        self.fail('Should raise error')