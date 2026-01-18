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
def test_per_subplot_kw_expander(self):
    normalize = Figure._norm_per_subplot_kw
    assert normalize({'A': {}, 'B': {}}) == {'A': {}, 'B': {}}
    assert normalize({('A', 'B'): {}}) == {'A': {}, 'B': {}}
    with pytest.raises(ValueError, match=f'The key {'B'!r} appears multiple times'):
        normalize({('A', 'B'): {}, 'B': {}})
    with pytest.raises(ValueError, match=f'The key {'B'!r} appears multiple times'):
        normalize({'B': {}, ('A', 'B'): {}})