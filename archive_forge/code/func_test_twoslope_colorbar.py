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
@image_comparison(['colorbar_twoslope.png'], remove_text=True, style='mpl20')
def test_twoslope_colorbar():
    fig, ax = plt.subplots()
    norm = mcolors.TwoSlopeNorm(20, 5, 95)
    pc = ax.pcolormesh(np.arange(1, 11), np.arange(1, 11), np.arange(100).reshape(10, 10), norm=norm, cmap='RdBu_r')
    fig.colorbar(pc)