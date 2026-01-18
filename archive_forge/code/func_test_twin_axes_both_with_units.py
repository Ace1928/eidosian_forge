from itertools import product
import io
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cbook
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.testing.decorators import (
from mpl_toolkits.axes_grid1 import (
from mpl_toolkits.axes_grid1.anchored_artists import (
from mpl_toolkits.axes_grid1.axes_divider import (
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from mpl_toolkits.axes_grid1.inset_locator import (
import mpl_toolkits.axes_grid1.mpl_axes
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
def test_twin_axes_both_with_units():
    host = host_subplot(111)
    host.plot_date([0, 1, 2], [0, 1, 2], xdate=False, ydate=True)
    twin = host.twinx()
    twin.plot(['a', 'b', 'c'])
    assert host.get_yticklabels()[0].get_text() == '00:00:00'
    assert twin.get_yticklabels()[0].get_text() == 'a'