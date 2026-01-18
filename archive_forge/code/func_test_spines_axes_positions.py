import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.spines import Spines
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@image_comparison(['spines_axes_positions'])
def test_spines_axes_positions():
    fig = plt.figure()
    x = np.linspace(0, 2 * np.pi, 100)
    y = 2 * np.sin(x)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('centered spines')
    ax.plot(x, y)
    ax.spines.right.set_position(('axes', 0.1))
    ax.yaxis.set_ticks_position('right')
    ax.spines.top.set_position(('axes', 0.25))
    ax.xaxis.set_ticks_position('top')
    ax.spines.left.set_color('none')
    ax.spines.bottom.set_color('none')