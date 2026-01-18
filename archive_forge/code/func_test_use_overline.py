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
def test_use_overline(self):
    """
        Test the parameter use_overline
        """
    x = 1 - 0.01
    fx1 = '$\\mathdefault{1-10^{-2}}$'
    fx2 = '$\\mathdefault{\\overline{10^{-2}}}$'
    form = mticker.LogitFormatter(use_overline=False)
    assert form(x) == fx1
    form.use_overline(True)
    assert form(x) == fx2
    form.use_overline(False)
    assert form(x) == fx1