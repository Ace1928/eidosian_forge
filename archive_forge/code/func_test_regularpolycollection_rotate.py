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
@image_comparison(['regularpolycollection_rotate.png'], remove_text=True)
def test_regularpolycollection_rotate():
    xx, yy = np.mgrid[:10, :10]
    xy_points = np.transpose([xx.flatten(), yy.flatten()])
    rotations = np.linspace(0, 2 * np.pi, len(xy_points))
    fig, ax = plt.subplots()
    for xy, alpha in zip(xy_points, rotations):
        col = mcollections.RegularPolyCollection(4, sizes=(100,), rotation=alpha, offsets=[xy], offset_transform=ax.transData)
        ax.add_collection(col, autolim=True)
    ax.autoscale_view()