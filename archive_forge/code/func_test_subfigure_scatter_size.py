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
@image_comparison(['test_subfigure_scatter_size.png'], style='mpl20', remove_text=True)
def test_subfigure_scatter_size():
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[1])
    ax0.scatter([1, 2, 3], [1, 2, 3], s=30, marker='s')
    ax0.scatter([3, 4, 5], [1, 2, 3], s=[20, 30, 40], marker='s')
    sfig = fig.add_subfigure(gs[0])
    axs = sfig.subplots(1, 2)
    for ax in [ax0, axs[0]]:
        ax.scatter([1, 2, 3], [1, 2, 3], s=30, marker='s', color='r')
        ax.scatter([3, 4, 5], [1, 2, 3], s=[20, 30, 40], marker='s', color='g')