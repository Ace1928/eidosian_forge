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
@pytest.mark.parametrize('str_pattern', ['abc', 'cab', 'bca', 'cba', 'acb', 'bac'])
def test_user_order(self, str_pattern):
    fig = plt.figure()
    ax_dict = fig.subplot_mosaic(str_pattern)
    assert list(str_pattern) == list(ax_dict)
    assert list(fig.axes) == list(ax_dict.values())