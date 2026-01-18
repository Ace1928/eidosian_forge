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
@pytest.mark.parametrize('pkg', ['xcolor', 'chemformula'])
def test_usetex_packages(pkg):
    if not _has_tex_package(pkg):
        pytest.skip(f'{pkg} is not available')
    mpl.rcParams['text.usetex'] = True
    fig = plt.figure()
    text = fig.text(0.5, 0.5, 'Some text 0123456789')
    fig.canvas.draw()
    mpl.rcParams['text.latex.preamble'] = '\\PassOptionsToPackage{dvipsnames}{xcolor}\\usepackage{%s}' % pkg
    fig = plt.figure()
    text2 = fig.text(0.5, 0.5, 'Some text 0123456789')
    fig.canvas.draw()
    np.testing.assert_array_equal(text2.get_window_extent(), text.get_window_extent())