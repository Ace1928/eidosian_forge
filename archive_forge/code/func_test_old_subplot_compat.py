import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_old_subplot_compat():
    fig = plt.figure()
    assert isinstance(fig.add_subplot(), SubplotBase)
    assert not isinstance(fig.add_axes(rect=[0, 0, 1, 1]), SubplotBase)
    with pytest.raises(TypeError):
        Axes(fig, [0, 0, 1, 1], rect=[0, 0, 1, 1])