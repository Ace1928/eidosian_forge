import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
@image_comparison(['quiver_key_xy.png'], remove_text=True)
def test_quiver_key_xy():
    X = np.arange(8)
    Y = np.zeros(8)
    angles = X * (np.pi / 4)
    uv = np.exp(1j * angles)
    U = uv.real
    V = uv.imag
    fig, axs = plt.subplots(2)
    for ax, angle_str in zip(axs, ('uv', 'xy')):
        ax.set_xlim(-1, 8)
        ax.set_ylim(-0.2, 0.2)
        q = ax.quiver(X, Y, U, V, pivot='middle', units='xy', width=0.05, scale=2, scale_units='xy', angles=angle_str)
        for x, angle in zip((0.2, 0.5, 0.8), (0, 45, 90)):
            ax.quiverkey(q, X=x, Y=0.8, U=1, angle=angle, label='', color='b')