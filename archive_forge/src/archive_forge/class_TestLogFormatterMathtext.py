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
class TestLogFormatterMathtext:
    fmt = mticker.LogFormatterMathtext()
    test_data = [(0, 1, '$\\mathdefault{10^{0}}$'), (0, 0.01, '$\\mathdefault{10^{-2}}$'), (0, 100.0, '$\\mathdefault{10^{2}}$'), (3, 1, '$\\mathdefault{1}$'), (3, 0.01, '$\\mathdefault{0.01}$'), (3, 100.0, '$\\mathdefault{100}$'), (3, 0.001, '$\\mathdefault{10^{-3}}$'), (3, 1000.0, '$\\mathdefault{10^{3}}$')]

    @pytest.mark.parametrize('min_exponent, value, expected', test_data)
    def test_min_exponent(self, min_exponent, value, expected):
        with mpl.rc_context({'axes.formatter.min_exponent': min_exponent}):
            assert self.fmt(value) == expected