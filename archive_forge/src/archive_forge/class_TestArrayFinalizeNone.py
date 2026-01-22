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
class TestArrayFinalizeNone(_DeprecationTestCase):
    message = 'Setting __array_finalize__ = None'

    def test_use_none_is_deprecated(self):

        class NoFinalize(np.ndarray):
            __array_finalize__ = None
        self.assert_deprecated(lambda: np.array(1).view(NoFinalize))