import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_thetalim_valid_invalid():
    ax = plt.subplot(projection='polar')
    ax.set_thetalim(0, 2 * np.pi)
    ax.set_thetalim(thetamin=800, thetamax=440)
    with pytest.raises(ValueError, match='angle range must be less than a full circle'):
        ax.set_thetalim(0, 3 * np.pi)
    with pytest.raises(ValueError, match='angle range must be less than a full circle'):
        ax.set_thetalim(thetamin=800, thetamax=400)