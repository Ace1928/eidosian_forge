import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_widgetlock_zoompan():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.widgetlock(ax)
    tb = NavigationToolbar2(fig.canvas)
    tb.zoom()
    assert ax.get_navigate_mode() is None
    tb.pan()
    assert ax.get_navigate_mode() is None