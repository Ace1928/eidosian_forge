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
def test_using_all_default_major_steps(self):
    with mpl.rc_context({'_internal.classic_mode': False}):
        majorsteps = [x[0] for x in self.majorstep_minordivisions]
        np.testing.assert_allclose(majorsteps, mticker.AutoLocator()._steps)