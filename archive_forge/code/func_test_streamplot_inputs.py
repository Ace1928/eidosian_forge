import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.transforms as mtransforms
def test_streamplot_inputs():
    plt.streamplot(np.arange(3), np.arange(3), np.full((3, 3), np.nan), np.full((3, 3), np.nan), color=np.random.rand(3, 3))
    plt.streamplot(range(3), range(3), np.random.rand(3, 3), np.random.rand(3, 3))