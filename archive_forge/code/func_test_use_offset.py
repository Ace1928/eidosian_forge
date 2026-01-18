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
@pytest.mark.parametrize('use_offset', use_offset_data)
def test_use_offset(self, use_offset):
    with mpl.rc_context({'axes.formatter.useoffset': use_offset}):
        tmp_form = mticker.ScalarFormatter()
        assert use_offset == tmp_form.get_useOffset()
        assert tmp_form.offset == 0