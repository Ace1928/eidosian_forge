from collections import Counter
from pathlib import Path
import io
import re
import tempfile
import numpy as np
import pytest
from matplotlib import cbook, path, patheffects, font_manager as fm
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
@image_comparison(['scatter.eps'])
def test_path_collection():
    rng = np.random.default_rng(19680801)
    xvals = rng.uniform(0, 1, 10)
    yvals = rng.uniform(0, 1, 10)
    sizes = rng.uniform(30, 100, 10)
    fig, ax = plt.subplots()
    ax.scatter(xvals, yvals, sizes, edgecolor=[0.9, 0.2, 0.1], marker='<')
    ax.set_axis_off()
    paths = [path.Path.unit_regular_polygon(i) for i in range(3, 7)]
    offsets = rng.uniform(0, 200, 20).reshape(10, 2)
    sizes = [0.02, 0.04]
    pc = mcollections.PathCollection(paths, sizes, zorder=-1, facecolors='yellow', offsets=offsets)
    ax.add_collection(pc)
    ax.set_xlim(0, 1)