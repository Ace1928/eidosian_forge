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
@image_comparison(['nonorm_colorbars.svg'], style='mpl20')
def test_nonorm():
    plt.rcParams['svg.fonttype'] = 'none'
    data = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = NoNorm(vmin=min(data), vmax=max(data))
    cmap = mpl.colormaps['viridis'].resampled(len(data))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, cax=ax, orientation='horizontal')