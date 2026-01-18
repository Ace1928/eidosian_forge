import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_pickle_2(self):
    a = self.data
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        assert_equal(a, pickle.loads(pickle.dumps(a, protocol=proto)))
        assert_equal(a[0], pickle.loads(pickle.dumps(a[0], protocol=proto)))