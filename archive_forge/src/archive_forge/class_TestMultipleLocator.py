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
class TestMultipleLocator:

    def test_basic(self):
        loc = mticker.MultipleLocator(base=3.147)
        test_value = np.array([-9.441, -6.294, -3.147, 0.0, 3.147, 6.294, 9.441, 12.588])
        assert_almost_equal(loc.tick_values(-7, 10), test_value)

    def test_basic_with_offset(self):
        loc = mticker.MultipleLocator(base=3.147, offset=1.2)
        test_value = np.array([-8.241, -5.094, -1.947, 1.2, 4.347, 7.494, 10.641])
        assert_almost_equal(loc.tick_values(-7, 10), test_value)

    def test_view_limits(self):
        """
        Test basic behavior of view limits.
        """
        with mpl.rc_context({'axes.autolimit_mode': 'data'}):
            loc = mticker.MultipleLocator(base=3.147)
            assert_almost_equal(loc.view_limits(-5, 5), (-5, 5))

    def test_view_limits_round_numbers(self):
        """
        Test that everything works properly with 'round_numbers' for auto
        limit.
        """
        with mpl.rc_context({'axes.autolimit_mode': 'round_numbers'}):
            loc = mticker.MultipleLocator(base=3.147)
            assert_almost_equal(loc.view_limits(-4, 4), (-6.294, 6.294))

    def test_view_limits_round_numbers_with_offset(self):
        """
        Test that everything works properly with 'round_numbers' for auto
        limit.
        """
        with mpl.rc_context({'axes.autolimit_mode': 'round_numbers'}):
            loc = mticker.MultipleLocator(base=3.147, offset=1.3)
            assert_almost_equal(loc.view_limits(-4, 4), (-4.994, 4.447))

    def test_set_params(self):
        """
        Create multiple locator with 0.7 base, and change it to something else.
        See if change was successful.
        """
        mult = mticker.MultipleLocator(base=0.7)
        mult.set_params(base=1.7)
        assert mult._edge.step == 1.7
        mult.set_params(offset=3)
        assert mult._offset == 3