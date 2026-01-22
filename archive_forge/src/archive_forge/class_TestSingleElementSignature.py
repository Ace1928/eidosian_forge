import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
class TestSingleElementSignature(_DeprecationTestCase):
    message = 'The use of a length 1'

    def test_deprecated(self):
        self.assert_deprecated(lambda: np.add(1, 2, signature='d'))
        self.assert_deprecated(lambda: np.add(1, 2, sig=(np.dtype('l'),)))