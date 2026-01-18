import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@image_comparison(['polar_coords'], style='default', remove_text=True, tol=0.012)
def test_polar_coord_annotations():
    el = mpl.patches.Ellipse((0, 0), 10, 20, facecolor='r', alpha=0.5)
    fig = plt.figure()
    ax = fig.add_subplot(aspect='equal')
    ax.add_artist(el)
    el.set_clip_box(ax.bbox)
    ax.annotate('the top', xy=(np.pi / 2.0, 10.0), xytext=(np.pi / 3, 20.0), xycoords='polar', textcoords='polar', arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='baseline', clip_on=True)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)