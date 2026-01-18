import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@image_comparison(['subplots_offset_text'])
def test_subplots_offsettext():
    x = np.arange(0, 10000000000.0, 1000000000.0)
    y = np.arange(0, 100, 10) + 10000.0
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='all')
    axs[0, 0].plot(x, x)
    axs[1, 0].plot(x, x)
    axs[0, 1].plot(y, x)
    axs[1, 1].plot(y, x)