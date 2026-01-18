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
def test_title_text_loc():
    plt.style.use('mpl20')
    fig, ax = plt.subplots()
    np.random.seed(seed=19680808)
    pc = ax.pcolormesh(np.random.randn(10, 10))
    cb = fig.colorbar(pc, location='right', extend='max')
    cb.ax.set_title('Aardvark')
    fig.draw_without_rendering()
    assert cb.ax.title.get_window_extent(fig.canvas.get_renderer()).ymax > cb.ax.spines['outline'].get_window_extent().ymax