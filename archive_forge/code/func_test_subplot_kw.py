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
@check_figures_equal(extensions=['png'])
@pytest.mark.parametrize('subplot_kw', [{}, {'projection': 'polar'}, None])
def test_subplot_kw(self, fig_test, fig_ref, subplot_kw):
    x = [[1, 2]]
    grid_axes = fig_test.subplot_mosaic(x, subplot_kw=subplot_kw)
    subplot_kw = subplot_kw or {}
    gs = fig_ref.add_gridspec(1, 2)
    axA = fig_ref.add_subplot(gs[0, 0], **subplot_kw)
    axB = fig_ref.add_subplot(gs[0, 1], **subplot_kw)