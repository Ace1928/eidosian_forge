import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_triplot_label():
    x = [0, 2, 1]
    y = [0, 0, 1]
    data = [[0, 1, 2]]
    fig, ax = plt.subplots()
    lines, markers = ax.triplot(x, y, data, label='label')
    handles, labels = ax.get_legend_handles_labels()
    assert labels == ['label']
    assert len(handles) == 1
    assert handles[0] is lines