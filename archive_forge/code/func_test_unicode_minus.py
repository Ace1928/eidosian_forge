from tempfile import TemporaryFile
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
from matplotlib import dviread
from matplotlib.testing import _has_tex_package
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
@check_figures_equal()
def test_unicode_minus(fig_test, fig_ref):
    mpl.rcParams['text.usetex'] = True
    fig_test.text(0.5, 0.5, '$-$')
    fig_ref.text(0.5, 0.5, '−')