import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_tripcolor_warnings():
    x = [-1, 0, 1, 0]
    y = [0, -1, 0, 1]
    c = [0.4, 0.5]
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match='Positional parameter c .*no effect'):
        ax.tripcolor(x, y, c, facecolors=c)
    with pytest.warns(UserWarning, match='Positional parameter c .*no effect'):
        ax.tripcolor(x, y, 'interpreted as c', facecolors=c)