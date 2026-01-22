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
class TestFormatStrFormatter:

    def test_basic(self):
        tmp_form = mticker.FormatStrFormatter('%05d')
        assert '00002' == tmp_form(2)