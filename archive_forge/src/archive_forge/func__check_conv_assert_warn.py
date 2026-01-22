import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
def _check_conv_assert_warn(self, val, expected):
    if self.warn:
        with assert_warns(DeprecationWarning) as exc:
            assert self.conv(val) == expected
    else:
        assert self.conv(val) == expected