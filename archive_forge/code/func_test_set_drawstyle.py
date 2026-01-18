import itertools
import platform
import timeit
from types import SimpleNamespace
from cycler import cycler
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import _path
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_set_drawstyle():
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)
    fig, ax = plt.subplots()
    line, = ax.plot(x, y)
    line.set_drawstyle('steps-pre')
    assert len(line.get_path().vertices) == 2 * len(x) - 1
    line.set_drawstyle('default')
    assert len(line.get_path().vertices) == len(x)