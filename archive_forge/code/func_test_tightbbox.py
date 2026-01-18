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
def test_tightbbox():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    t = ax.text(1.0, 0.5, 'This dangles over end')
    renderer = fig.canvas.get_renderer()
    x1Nom0 = 9.035
    assert abs(t.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom0) < 0.05
    assert abs(fig.get_tightbbox(renderer).x0 - 0.679) < 0.05
    t.set_in_layout(False)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom * fig.dpi) < 2
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom) < 0.05
    t.set_in_layout(True)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    assert abs(ax.get_tightbbox(renderer, bbox_extra_artists=[]).x1 - x1Nom * fig.dpi) < 2