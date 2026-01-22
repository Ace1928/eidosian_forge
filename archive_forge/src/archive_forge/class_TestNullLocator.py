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
class TestNullLocator:

    def test_set_params(self):
        """
        Create null locator, and attempt to call set_params() on it.
        Should not exception, and should raise a warning.
        """
        loc = mticker.NullLocator()
        with pytest.warns(UserWarning):
            loc.set_params()