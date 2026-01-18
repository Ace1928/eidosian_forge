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
def test_void_getitem(self):
    assert_(np.array([b'a'], 'V1').astype('O') == b'a')
    assert_(np.array([b'ab'], 'V2').astype('O') == b'ab')
    assert_(np.array([b'abc'], 'V3').astype('O') == b'abc')
    assert_(np.array([b'abcd'], 'V4').astype('O') == b'abcd')