from collections import Counter
from pathlib import Path
import io
import re
import tempfile
import numpy as np
import pytest
from matplotlib import cbook, path, patheffects, font_manager as fm
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
@check_figures_equal(extensions=['eps'])
def test_text_clip(fig_test, fig_ref):
    ax = fig_test.add_subplot()
    ax.text(0, 0, 'hello', transform=fig_test.transFigure, clip_on=True)
    fig_ref.add_subplot()