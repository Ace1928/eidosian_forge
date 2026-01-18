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
def test_reused_gridspec():
    """Test that these all use the same gridspec"""
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, (3, 5))
    ax2 = fig.add_subplot(3, 2, 4)
    ax3 = plt.subplot2grid((3, 2), (2, 1), colspan=2, fig=fig)
    gs1 = ax1.get_subplotspec().get_gridspec()
    gs2 = ax2.get_subplotspec().get_gridspec()
    gs3 = ax3.get_subplotspec().get_gridspec()
    assert gs1 == gs2
    assert gs1 == gs3