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
@needs_usetex
def test_partial_usetex(caplog):
    caplog.set_level('WARNING')
    plt.figtext(0.1, 0.1, 'foo', usetex=True)
    plt.figtext(0.2, 0.2, 'bar', usetex=True)
    plt.savefig(io.BytesIO(), format='ps')
    record, = caplog.records
    assert 'as if usetex=False' in record.getMessage()