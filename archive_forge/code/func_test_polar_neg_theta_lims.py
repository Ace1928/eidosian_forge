import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_polar_neg_theta_lims():
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.set_thetalim(-np.pi, np.pi)
    labels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    assert labels == ['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°']