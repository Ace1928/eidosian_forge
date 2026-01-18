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
@pytest.mark.xfail(pyparsing_version.release == (3, 1, 0), reason='Error messages are incorrect with pyparsing 3.1.0')
def test_parse_math():
    fig, ax = plt.subplots()
    ax.text(0, 0, '$ \\wrong{math} $', parse_math=False)
    fig.canvas.draw()
    ax.text(0, 0, '$ \\wrong{math} $', parse_math=True)
    with pytest.raises(ValueError, match='Unknown symbol'):
        fig.canvas.draw()