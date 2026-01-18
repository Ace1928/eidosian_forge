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
@image_comparison(baseline_images=['test_usetex'], extensions=['pdf', 'png'], style='mpl20')
def test_usetex():
    mpl.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    kwargs = {'verticalalignment': 'baseline', 'size': 24, 'bbox': dict(pad=0, edgecolor='k', facecolor='none')}
    ax.text(0.2, 0.7, '\\LaTeX\\ $\\left[\\int\\limits_e^{2e}\\sqrt\\frac{\\log^3 x}{x}\\,\\mathrm{d}x \\right\\}$', **kwargs)
    ax.text(0.2, 0.3, 'lg', **kwargs)
    ax.text(0.4, 0.3, '$\\frac{1}{2}\\pi$', **kwargs)
    ax.text(0.6, 0.3, '$p^{3^A}$', **kwargs)
    ax.text(0.8, 0.3, '$p_{3_2}$', **kwargs)
    for x in {t.get_position()[0] for t in ax.texts}:
        ax.axvline(x)
    for y in {t.get_position()[1] for t in ax.texts}:
        ax.axhline(y)
    ax.set_axis_off()