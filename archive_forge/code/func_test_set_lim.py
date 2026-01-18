import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
def test_set_lim():
    f, ax = plt.subplots()
    ax.plot(['a', 'b', 'c', 'd'], [1, 2, 3, 4])
    with warnings.catch_warnings():
        ax.set_xlim('b', 'c')