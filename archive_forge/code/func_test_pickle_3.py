import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_pickle_3(self):
    a = self.data
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        pa = pickle.loads(pickle.dumps(a[0], protocol=proto))
        assert_(pa.flags.c_contiguous)
        assert_(pa.flags.f_contiguous)
        assert_(pa.flags.writeable)
        assert_(pa.flags.aligned)