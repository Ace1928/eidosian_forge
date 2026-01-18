import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
@image_comparison(['quiver_animated_test_image.png'])
def test_quiver_animate():
    fig, ax = plt.subplots()
    Q = draw_quiver(ax, animated=True)
    ax.quiverkey(Q, 0.5, 0.92, 2, '$2 \\frac{m}{s}$', labelpos='W', fontproperties={'weight': 'bold'})