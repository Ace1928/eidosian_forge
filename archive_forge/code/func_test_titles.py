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
@image_comparison(['titles'])
def test_titles():
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title('left title', loc='left')
    ax.set_title('right title', loc='right')
    ax.set_xticks([])
    ax.set_yticks([])