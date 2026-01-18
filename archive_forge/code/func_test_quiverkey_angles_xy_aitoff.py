import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_quiverkey_angles_xy_aitoff():
    kwargs_list = [{'angles': 'xy'}, {'angles': 'xy', 'scale_units': 'xy'}, {'scale_units': 'xy'}]
    for kwargs_dict in kwargs_list:
        x = np.linspace(-np.pi, np.pi, 11)
        y = np.ones_like(x) * np.pi / 6
        vx = np.zeros_like(x)
        vy = np.ones_like(x)
        fig = plt.figure()
        ax = fig.add_subplot(projection='aitoff')
        q = ax.quiver(x, y, vx, vy, **kwargs_dict)
        qk = ax.quiverkey(q, 0, 0, 1, '1 units')
        fig.canvas.draw()
        assert len(qk.verts) == 1