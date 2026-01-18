import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal(extensions=['png'])
def test_polar_interpolation_steps_variable_r(fig_test, fig_ref):
    l, = fig_test.add_subplot(projection='polar').plot([0, np.pi / 2], [1, 2])
    l.get_path()._interpolation_steps = 100
    fig_ref.add_subplot(projection='polar').plot(np.linspace(0, np.pi / 2, 101), np.linspace(1, 2, 101))