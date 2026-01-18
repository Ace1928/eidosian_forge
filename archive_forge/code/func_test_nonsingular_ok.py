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
@pytest.mark.parametrize('lims', [(a, b) for a, b in itertools.product(acceptable_vmin_vmax, repeat=2) if a != b])
def test_nonsingular_ok(self, lims):
    """
        Create logit locator, and test the nonsingular method for acceptable
        value
        """
    loc = mticker.LogitLocator()
    lims2 = loc.nonsingular(*lims)
    assert sorted(lims) == sorted(lims2)