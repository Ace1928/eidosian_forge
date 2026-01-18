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
def test_set_params(self):
    lctr = mticker.AsinhLocator(linear_width=5, numticks=17, symthresh=0.125, base=4, subs=(2.5, 3.25))
    assert lctr.numticks == 17
    assert lctr.symthresh == 0.125
    assert lctr.base == 4
    assert lctr.subs == (2.5, 3.25)
    lctr.set_params(numticks=23)
    assert lctr.numticks == 23
    lctr.set_params(None)
    assert lctr.numticks == 23
    lctr.set_params(symthresh=0.5)
    assert lctr.symthresh == 0.5
    lctr.set_params(symthresh=None)
    assert lctr.symthresh == 0.5
    lctr.set_params(base=7)
    assert lctr.base == 7
    lctr.set_params(base=None)
    assert lctr.base == 7
    lctr.set_params(subs=(2, 4.125))
    assert lctr.subs == (2, 4.125)
    lctr.set_params(subs=None)
    assert lctr.subs == (2, 4.125)
    lctr.set_params(subs=[])
    assert lctr.subs is None