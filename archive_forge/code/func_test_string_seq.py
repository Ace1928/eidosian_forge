from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
def test_string_seq(self):
    a1 = ['a', 'b', 'c', 'd', 'e', 'f']
    a2 = [1, 2, 3, np.nan, np.nan, 6]
    result1, result2 = delete_masked_points(a1, a2)
    ind = [0, 1, 2, 5]
    assert_array_equal(result1, np.array(a1)[ind])
    assert_array_equal(result2, np.array(a2)[ind])