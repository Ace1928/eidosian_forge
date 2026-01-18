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
@image_comparison(['size_in_xy.png'], remove_text=True)
def test_size_in_xy():
    fig, ax = plt.subplots()
    widths, heights, angles = ((10, 10), 10, 0)
    widths = (10, 10)
    coords = [(10, 10), (15, 15)]
    e = mcollections.EllipseCollection(widths, heights, angles, units='xy', offsets=coords, offset_transform=ax.transData)
    ax.add_collection(e)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)