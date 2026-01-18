import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
@check_figures_equal(extensions=['png'])
def test_patch_linestyle_none(fig_test, fig_ref):
    circle = mpath.Path.unit_circle()
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()
    for i, ls in enumerate(['none', 'None', ' ', '']):
        path = mpath.Path(circle.vertices + i, circle.codes)
        patch = mpatches.PathPatch(path, linewidth=3, linestyle=ls, facecolor=(1, 0, 0), edgecolor=(0, 0, 1))
        ax_test.add_patch(patch)
        patch = mpatches.PathPatch(path, linewidth=3, linestyle='-', facecolor=(1, 0, 0), edgecolor='none')
        ax_ref.add_patch(patch)
    ax_test.set_xlim([-1, i + 1])
    ax_test.set_ylim([-1, i + 1])
    ax_ref.set_xlim([-1, i + 1])
    ax_ref.set_ylim([-1, i + 1])