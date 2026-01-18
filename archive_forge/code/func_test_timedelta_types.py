import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def test_timedelta_types(self):
    """
        Test from_dtype() and as_dtype() with the timedelta types.
        """
    self.check_datetime_types('m', types.NPTimedelta)