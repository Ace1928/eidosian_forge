from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_set_offset_units():
    x = np.linspace(0, 10, 5)
    y = np.sin(x)
    d = x * np.timedelta64(24, 'h') + np.datetime64('2021-11-29')
    sc = plt.scatter(d, y)
    off0 = sc.get_offsets()
    sc.set_offsets(list(zip(d, y)))
    np.testing.assert_allclose(off0, sc.get_offsets())
    fig, ax = plt.subplots()
    sc = ax.scatter(y, d)
    off0 = sc.get_offsets()
    sc.set_offsets(list(zip(y, d)))
    np.testing.assert_allclose(off0, sc.get_offsets())