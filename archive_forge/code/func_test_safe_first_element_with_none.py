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
def test_safe_first_element_with_none():
    datetime_lst = [date.today() + timedelta(days=i) for i in range(10)]
    datetime_lst[0] = None
    actual = cbook._safe_first_finite(datetime_lst)
    assert actual is not None and actual == datetime_lst[1]