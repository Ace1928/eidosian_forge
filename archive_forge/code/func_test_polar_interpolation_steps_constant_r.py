import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal(extensions=['png'])
def test_polar_interpolation_steps_constant_r(fig_test, fig_ref):
    p1 = fig_test.add_subplot(121, projection='polar').bar([0], [1], 3 * np.pi, edgecolor='none', antialiased=False)
    p2 = fig_test.add_subplot(122, projection='polar').bar([0], [1], -3 * np.pi, edgecolor='none', antialiased=False)
    p3 = fig_ref.add_subplot(121, projection='polar').bar([0], [1], 2 * np.pi, edgecolor='none', antialiased=False)
    p4 = fig_ref.add_subplot(122, projection='polar').bar([0], [1], -2 * np.pi, edgecolor='none', antialiased=False)