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
class TestLogFormatterExponent:
    param_data = [(True, 4, np.arange(-3, 4.0), np.arange(-3, 4.0), ['-3', '-2', '-1', '0', '1', '2', '3']), (False, 10, np.array([0.1, 1e-05, np.pi, 0.2, -0.2, -1e-05]), range(6), ['0.1', '1e-05', '3.14', '0.2', '-0.2', '-1e-05']), (False, 50, np.array([3, 5, 12, 42], dtype=float), range(6), ['3', '5', '12', '42'])]
    base_data = [2.0, 5.0, 10.0, np.pi, np.e]

    @pytest.mark.parametrize('labelOnlyBase, exponent, locs, positions, expected', param_data)
    @pytest.mark.parametrize('base', base_data)
    def test_basic(self, labelOnlyBase, base, exponent, locs, positions, expected):
        formatter = mticker.LogFormatterExponent(base=base, labelOnlyBase=labelOnlyBase)
        formatter.create_dummy_axis()
        formatter.axis.set_view_interval(1, base ** exponent)
        vals = base ** locs
        labels = [formatter(x, pos) for x, pos in zip(vals, positions)]
        expected = [label.replace('-', '−') for label in expected]
        assert labels == expected

    def test_blank(self):
        formatter = mticker.LogFormatterExponent(base=10, labelOnlyBase=True)
        formatter.create_dummy_axis()
        formatter.axis.set_view_interval(1, 10)
        assert formatter(10 ** 0.1) == ''