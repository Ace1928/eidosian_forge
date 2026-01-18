import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
@image_comparison(['quiver_with_key_test_image.png'])
def test_quiver_with_key():
    fig, ax = plt.subplots()
    ax.margins(0.1)
    Q = draw_quiver(ax)
    ax.quiverkey(Q, 0.5, 0.95, 2, '$2\\, \\mathrm{m}\\, \\mathrm{s}^{-1}$', angle=-10, coordinates='figure', labelpos='W', fontproperties={'weight': 'bold', 'size': 'large'})