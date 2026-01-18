import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
@image_comparison(['proportional_colorbars.png'], remove_text=True, style='mpl20')
def test_proportional_colorbars():
    x = y = np.arange(-3.0, 3.01, 0.025)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X ** 2 - Y ** 2)
    Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2
    levels = [-1.25, -0.5, -0.125, 0.125, 0.5, 1.25]
    cmap = mcolors.ListedColormap(['0.3', '0.5', 'white', 'lightblue', 'steelblue'])
    cmap.set_under('darkred')
    cmap.set_over('crimson')
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    extends = ['neither', 'both']
    spacings = ['uniform', 'proportional']
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            CS3 = axs[i, j].contourf(X, Y, Z, levels, cmap=cmap, norm=norm, extend=extends[i])
            fig.colorbar(CS3, spacing=spacings[j], ax=axs[i, j])