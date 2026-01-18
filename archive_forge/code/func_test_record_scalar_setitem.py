import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_record_scalar_setitem(self):
    rec = np.recarray(1, dtype=[('x', float, 5)])
    rec[0].x = 1
    assert_equal(rec[0].x, np.ones(5))