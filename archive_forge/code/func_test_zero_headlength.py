import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_zero_headlength():
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    U, V = (np.cos(X), np.sin(Y))
    ax.quiver(U, V, headlength=0, headaxislength=0)
    fig.canvas.draw()