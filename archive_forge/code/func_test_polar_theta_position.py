import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@image_comparison(['polar_theta_position'], style='default')
def test_polar_theta_position():
    r = np.arange(0, 3.0, 0.01)
    theta = 2 * np.pi * r
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.plot(theta, r)
    ax.set_theta_zero_location('NW', 30)
    ax.set_theta_direction('clockwise')