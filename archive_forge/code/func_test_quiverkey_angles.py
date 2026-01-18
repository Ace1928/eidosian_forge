import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_quiverkey_angles():
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(2), np.arange(2))
    U = V = angles = np.ones_like(X)
    q = ax.quiver(X, Y, U, V, angles=angles)
    qk = ax.quiverkey(q, 1, 1, 2, 'Label')
    fig.canvas.draw()
    assert len(qk.verts) == 1