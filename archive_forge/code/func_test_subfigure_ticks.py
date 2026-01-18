import copy
from datetime import datetime
import io
from pathlib import Path
import pickle
import platform
from threading import Timer
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from PIL import Image
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, FigureBase
from matplotlib.layout_engine import (ConstrainedLayoutEngine,
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
@mpl.style.context('mpl20')
def test_subfigure_ticks():
    fig = plt.figure(constrained_layout=True, figsize=(10, 3))
    subfig_bl, subfig_br = fig.subfigures(1, 2, wspace=0.01, width_ratios=[7, 2])
    gs = subfig_bl.add_gridspec(nrows=1, ncols=14)
    ax1 = subfig_bl.add_subplot(gs[0, :1])
    ax1.scatter(x=[-56.46881504821776, 24.179891162109396], y=[1500, 3600])
    ax2 = subfig_bl.add_subplot(gs[0, 1:3], sharey=ax1)
    ax2.scatter(x=[-126.5357270050049, 94.68456736755368], y=[1500, 3600])
    ax3 = subfig_bl.add_subplot(gs[0, 3:14], sharey=ax1)
    fig.dpi = 120
    fig.draw_without_rendering()
    ticks120 = ax2.get_xticks()
    fig.dpi = 300
    fig.draw_without_rendering()
    ticks300 = ax2.get_xticks()
    np.testing.assert_allclose(ticks120, ticks300)