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
class TestFixedLocator:

    def test_set_params(self):
        """
        Create fixed locator with 5 nbins, and change it to something else.
        See if change was successful.
        Should not exception.
        """
        fixed = mticker.FixedLocator(range(0, 24), nbins=5)
        fixed.set_params(nbins=7)
        assert fixed.nbins == 7