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
def test_waitforbuttonpress(recwarn):
    warnings.filterwarnings('ignore', 'cannot show the figure')
    fig = plt.figure()
    assert fig.waitforbuttonpress(timeout=0.1) is None
    Timer(0.1, KeyEvent('key_press_event', fig.canvas, 'z')._process).start()
    assert fig.waitforbuttonpress() is True
    Timer(0.1, MouseEvent('button_press_event', fig.canvas, 0, 0, 1)._process).start()
    assert fig.waitforbuttonpress() is False