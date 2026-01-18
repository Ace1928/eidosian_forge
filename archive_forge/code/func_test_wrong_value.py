import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
def test_wrong_value(self):
    self._check_value_error('')
    self._check_value_error('π')
    if self.allow_bytes:
        self._check_value_error(b'')
        self._check_value_error(b'\xff')
    if self.exact_match:
        self._check_value_error("there's no way this is supported")