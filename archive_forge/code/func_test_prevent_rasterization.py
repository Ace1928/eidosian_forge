import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
@check_figures_equal(tol=5, extensions=['svg', 'pdf'])
def test_prevent_rasterization(fig_test, fig_ref):
    loc = [0.05, 0.05]
    ax_ref = fig_ref.subplots()
    ax_ref.plot([loc[0]], [loc[1]], marker='x', c='black', zorder=2)
    b = mpl.offsetbox.TextArea('X')
    abox = mpl.offsetbox.AnnotationBbox(b, loc, zorder=2.1)
    ax_ref.add_artist(abox)
    ax_test = fig_test.subplots()
    ax_test.plot([loc[0]], [loc[1]], marker='x', c='black', zorder=2, rasterized=True)
    b = mpl.offsetbox.TextArea('X')
    abox = mpl.offsetbox.AnnotationBbox(b, loc, zorder=2.1)
    ax_test.add_artist(abox)