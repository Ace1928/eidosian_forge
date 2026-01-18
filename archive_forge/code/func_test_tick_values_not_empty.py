from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def test_tick_values_not_empty(self):
    mpl.rcParams['_internal.classic_mode'] = False
    ll = mticker.LogLocator(subs=(1, 2, 5))
    test_value = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0, 200000.0, 500000.0, 1000000.0, 2000000.0, 5000000.0, 10000000.0, 20000000.0, 50000000.0, 100000000.0, 200000000.0, 500000000.0, 1000000000.0, 2000000000.0, 5000000000.0])
    assert_almost_equal(ll.tick_values(1, 100000000.0), test_value)