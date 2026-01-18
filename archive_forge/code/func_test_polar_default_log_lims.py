import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_polar_default_log_lims():
    plt.subplot(projection='polar')
    ax = plt.gca()
    ax.set_rscale('log')
    assert ax.get_rmin() > 0