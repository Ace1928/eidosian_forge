import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@image_comparison(['tri_smooth_contouring.png'], remove_text=True, tol=0.072)
def test_tri_smooth_contouring():
    n_angles = 20
    n_radii = 10
    min_radius = 0.15

    def z(x, y):
        r1 = np.hypot(0.5 - x, 0.5 - y)
        theta1 = np.arctan2(0.5 - x, 0.5 - y)
        r2 = np.hypot(-x - 0.2, -y - 0.2)
        theta2 = np.arctan2(-x - 0.2, -y - 0.2)
        z = -(2 * (np.exp((r1 / 10) ** 2) - 1) * 30.0 * np.cos(7.0 * theta1) + (np.exp((r2 / 10) ** 2) - 1) * 30.0 * np.cos(11.0 * theta2) + 0.7 * (x ** 2 + y ** 2))
        return (np.max(z) - z) / (np.max(z) - np.min(z))
    radii = np.linspace(min_radius, 0.95, n_radii)
    angles = np.linspace(0 + n_angles, 2 * np.pi + n_angles, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles
    x0 = (radii * np.cos(angles)).flatten()
    y0 = (radii * np.sin(angles)).flatten()
    triang0 = mtri.Triangulation(x0, y0)
    z0 = z(x0, y0)
    triang0.set_mask(np.hypot(x0[triang0.triangles].mean(axis=1), y0[triang0.triangles].mean(axis=1)) < min_radius)
    refiner = mtri.UniformTriRefiner(triang0)
    tri_refi, z_test_refi = refiner.refine_field(z0, subdiv=4)
    levels = np.arange(0.0, 1.0, 0.025)
    plt.triplot(triang0, lw=0.5, color='0.5')
    plt.tricontour(tri_refi, z_test_refi, levels=levels, colors='black')