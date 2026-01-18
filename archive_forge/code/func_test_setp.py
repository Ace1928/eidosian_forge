import io
from itertools import chain
import numpy as np
import pytest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_setp():
    plt.setp([])
    plt.setp([[]])
    fig, ax = plt.subplots()
    lines1 = ax.plot(range(3))
    lines2 = ax.plot(range(3))
    martist.setp(chain(lines1, lines2), 'lw', 5)
    plt.setp(ax.spines.values(), color='green')
    sio = io.StringIO()
    plt.setp(lines1, 'zorder', file=sio)
    assert sio.getvalue() == '  zorder: float\n'