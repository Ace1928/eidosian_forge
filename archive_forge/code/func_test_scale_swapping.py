import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal(extensions=['png'])
def test_scale_swapping(fig_test, fig_ref):
    np.random.seed(19680801)
    samples = np.random.normal(size=10)
    x = np.linspace(-5, 5, 10)
    for fig, log_state in zip([fig_test, fig_ref], [True, False]):
        ax = fig.subplots()
        ax.hist(samples, log=log_state, density=True)
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi))
        fig.canvas.draw()
        ax.set_yscale('linear')