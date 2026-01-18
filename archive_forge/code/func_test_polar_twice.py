import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_polar_twice():
    fig = plt.figure()
    plt.polar([1, 2], [0.1, 0.2])
    plt.polar([3, 4], [0.3, 0.4])
    assert len(fig.axes) == 1, 'More than one polar axes created.'