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
def test_subfigure_spanning():
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    sub_figs = [fig.add_subfigure(gs[0, 0]), fig.add_subfigure(gs[0:2, 1]), fig.add_subfigure(gs[2, 1:3]), fig.add_subfigure(gs[0:, 1:])]
    w = 640
    h = 480
    np.testing.assert_allclose(sub_figs[0].bbox.min, [0.0, h * 2 / 3])
    np.testing.assert_allclose(sub_figs[0].bbox.max, [w / 3, h])
    np.testing.assert_allclose(sub_figs[1].bbox.min, [w / 3, h / 3])
    np.testing.assert_allclose(sub_figs[1].bbox.max, [w * 2 / 3, h])
    np.testing.assert_allclose(sub_figs[2].bbox.min, [w / 3, 0])
    np.testing.assert_allclose(sub_figs[2].bbox.max, [w, h / 3])
    for i in range(4):
        sub_figs[i].add_subplot()
    fig.draw_without_rendering()