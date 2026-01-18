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
def test_set_fig_size():
    fig = plt.figure()
    fig.set_figwidth(5)
    assert fig.get_figwidth() == 5
    fig.set_figheight(1)
    assert fig.get_figheight() == 1
    fig.set_size_inches(2, 4)
    assert fig.get_figwidth() == 2
    assert fig.get_figheight() == 4
    fig.set_size_inches((1, 3))
    assert fig.get_figwidth() == 1
    assert fig.get_figheight() == 3