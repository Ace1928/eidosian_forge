import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal()
def test_triplot_with_ls(fig_test, fig_ref):
    x = [0, 2, 1]
    y = [0, 0, 1]
    data = [[0, 1, 2]]
    fig_test.subplots().triplot(x, y, data, ls='--')
    fig_ref.subplots().triplot(x, y, data, linestyle='--')