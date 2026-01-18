import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal(extensions=['png'])
def test_remove_shared_polar(fig_ref, fig_test):
    axs = fig_ref.subplots(2, 2, sharex=True, subplot_kw={'projection': 'polar'})
    for i in [0, 1, 3]:
        axs.flat[i].remove()
    axs = fig_test.subplots(2, 2, sharey=True, subplot_kw={'projection': 'polar'})
    for i in [0, 1, 3]:
        axs.flat[i].remove()