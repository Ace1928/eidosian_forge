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
def test_pickle_string_overwrite(self):
    import re
    data = np.array([1], dtype='b')
    blob = pickle.dumps(data, protocol=1)
    data = pickle.loads(blob)
    s = re.sub('a(.)', '\x01\\1', 'a_')
    assert_equal(s[0], '\x01')
    data[0] = 106
    s = re.sub('a(.)', '\x01\\1', 'a_')
    assert_equal(s[0], '\x01')