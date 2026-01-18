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
def test_results_whiskers_float(self):
    results = cbook.boxplot_stats(self.data, whis=3)
    res = results[0]
    for key, value in self.known_whis3_res.items():
        assert_array_almost_equal(res[key], value)