import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_sign_for_complex_nan(self):
    with np.errstate(invalid='ignore'):
        C = np.array([-np.inf, -2 + 1j, 0, 2 - 1j, np.inf, np.nan])
        have = np.sign(C)
        want = np.array([-1 + 0j, -1 + 0j, 0 + 0j, 1 + 0j, 1 + 0j, np.nan])
        assert_equal(have, want)