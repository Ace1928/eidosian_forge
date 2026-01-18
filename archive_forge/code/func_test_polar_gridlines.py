import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_polar_gridlines():
    fig = plt.figure()
    ax = fig.add_subplot(polar=True)
    ax.grid(alpha=0.2)
    plt.setp(ax.yaxis.get_ticklabels(), visible=False)
    fig.canvas.draw()
    assert ax.xaxis.majorTicks[0].gridline.get_alpha() == 0.2
    assert ax.yaxis.majorTicks[0].gridline.get_alpha() == 0.2