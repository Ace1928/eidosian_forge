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
def test_view_limits(self):
    """
        Test basic behavior of view limits.
        """
    with mpl.rc_context({'axes.autolimit_mode': 'data'}):
        loc = mticker.MultipleLocator(base=3.147)
        assert_almost_equal(loc.view_limits(-5, 5), (-5, 5))