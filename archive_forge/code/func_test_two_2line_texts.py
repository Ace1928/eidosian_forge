from datetime import datetime
import io
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom
@pytest.mark.parametrize('spacing1, spacing2', [(0.4, 2), (2, 0.4), (2, 2)])
def test_two_2line_texts(spacing1, spacing2):
    text_string = 'line1\nline2'
    fig = plt.figure()
    renderer = fig.canvas.get_renderer()
    text1 = fig.text(0.25, 0.5, text_string, linespacing=spacing1)
    text2 = fig.text(0.25, 0.5, text_string, linespacing=spacing2)
    fig.canvas.draw()
    box1 = text1.get_window_extent(renderer=renderer)
    box2 = text2.get_window_extent(renderer=renderer)
    assert box1.width == box2.width
    if spacing1 == spacing2:
        assert box1.height == box2.height
    else:
        assert box1.height != box2.height