import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_trirefine():
    n = 3
    subdiv = 2
    x = np.linspace(-1.0, 1.0, n + 1)
    x, y = np.meshgrid(x, x)
    x = x.ravel()
    y = y.ravel()
    mask = np.zeros(2 * n ** 2, dtype=bool)
    mask[n ** 2:] = True
    triang = mtri.Triangulation(x, y, triangles=meshgrid_triangles(n + 1), mask=mask)
    refiner = mtri.UniformTriRefiner(triang)
    refi_triang = refiner.refine_triangulation(subdiv=subdiv)
    x_refi = refi_triang.x
    y_refi = refi_triang.y
    n_refi = n * subdiv ** 2
    x_verif = np.linspace(-1.0, 1.0, n_refi + 1)
    x_verif, y_verif = np.meshgrid(x_verif, x_verif)
    x_verif = x_verif.ravel()
    y_verif = y_verif.ravel()
    ind1d = np.isin(np.around(x_verif * (2.5 + y_verif), 8), np.around(x_refi * (2.5 + y_refi), 8))
    assert_array_equal(ind1d, True)
    refi_mask = refi_triang.mask
    refi_tri_barycenter_x = np.sum(refi_triang.x[refi_triang.triangles], axis=1) / 3.0
    refi_tri_barycenter_y = np.sum(refi_triang.y[refi_triang.triangles], axis=1) / 3.0
    tri_finder = triang.get_trifinder()
    refi_tri_indices = tri_finder(refi_tri_barycenter_x, refi_tri_barycenter_y)
    refi_tri_mask = triang.mask[refi_tri_indices]
    assert_array_equal(refi_mask, refi_tri_mask)
    x = np.asarray([0.0, 1.0, 0.0, 1.0])
    y = np.asarray([0.0, 0.0, 1.0, 1.0])
    triang = [mtri.Triangulation(x, y, [[0, 1, 3], [3, 2, 0]]), mtri.Triangulation(x, y, [[0, 1, 3], [2, 0, 3]])]
    z = np.hypot(x - 0.3, y - 0.4)
    xyz_data = []
    for i in range(2):
        refiner = mtri.UniformTriRefiner(triang[i])
        refined_triang, refined_z = refiner.refine_field(z, subdiv=1)
        xyz = np.dstack((refined_triang.x, refined_triang.y, refined_z))[0]
        xyz = xyz[np.lexsort((xyz[:, 1], xyz[:, 0]))]
        xyz_data += [xyz]
    assert_array_almost_equal(xyz_data[0], xyz_data[1])