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
def test_text_size_binding():
    mpl.rcParams['font.size'] = 10
    fp = mpl.font_manager.FontProperties(size='large')
    sz1 = fp.get_size_in_points()
    mpl.rcParams['font.size'] = 100
    assert sz1 == fp.get_size_in_points()