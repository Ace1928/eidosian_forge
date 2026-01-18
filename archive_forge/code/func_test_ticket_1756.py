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
def test_ticket_1756(self):
    s = b'0123456789abcdef'
    a = np.array([s] * 5)
    for i in range(1, 17):
        a1 = np.array(a, '|S%d' % i)
        a2 = np.array([s[:i]] * 5)
        assert_equal(a1, a2)