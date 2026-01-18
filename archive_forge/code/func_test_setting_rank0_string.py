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
def test_setting_rank0_string(self):
    """Ticket #1736"""
    s1 = b'hello1'
    s2 = b'hello2'
    a = np.zeros((), dtype='S10')
    a[()] = s1
    assert_equal(a, np.array(s1))
    a[()] = np.array(s2)
    assert_equal(a, np.array(s2))
    a = np.zeros((), dtype='f4')
    a[()] = 3
    assert_equal(a, np.array(3))
    a[()] = np.array(4)
    assert_equal(a, np.array(4))