import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_tripcolor_color():
    x = [-1, 0, 1, 0]
    y = [0, -1, 0, 1]
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match='tripcolor\\(\\) missing 1 required '):
        ax.tripcolor(x, y)
    with pytest.raises(ValueError, match='The length of c must match either'):
        ax.tripcolor(x, y, [1, 2, 3])
    with pytest.raises(ValueError, match='length of facecolors must match .* triangles'):
        ax.tripcolor(x, y, facecolors=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="'gouraud' .* at the points.* not at the faces"):
        ax.tripcolor(x, y, facecolors=[1, 2], shading='gouraud')
    with pytest.raises(ValueError, match="'gouraud' .* at the points.* not at the faces"):
        ax.tripcolor(x, y, [1, 2], shading='gouraud')
    with pytest.raises(TypeError, match="positional.*'c'.*keyword-only.*'facecolors'"):
        ax.tripcolor(x, y, C=[1, 2, 3, 4])
    with pytest.raises(TypeError, match='Unexpected positional parameter'):
        ax.tripcolor(x, y, [1, 2], 'unused_positional')
    ax.tripcolor(x, y, [1, 2, 3, 4])
    ax.tripcolor(x, y, [1, 2, 3, 4], shading='gouraud')
    ax.tripcolor(x, y, [1, 2])
    ax.tripcolor(x, y, facecolors=[1, 2])