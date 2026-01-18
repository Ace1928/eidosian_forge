import numpy as np
from holoviews.element import Dataset, Image, QuadMesh
from .test_plot import TestMPLPlot, mpl38, mpl_renderer
def test_quadmesh_update_cbar(self):
    xs = ys = np.linspace(0, 6, 10)
    zs = np.linspace(1, 2, 5)
    XS, _YS, ZS = np.meshgrid(xs, ys, zs)
    values = np.sin(XS) * ZS
    ds = Dataset((xs, ys, zs, values.T), ['x', 'y', 'z'], 'values')
    hmap = ds.to(QuadMesh).opts(colorbar=True, framewise=True)
    plot = mpl_renderer.get_plot(hmap)
    cbar = plot.handles['cbar']
    np.testing.assert_allclose([cbar.vmin, cbar.vmax], [-0.9989549170979283, 0.9719379013633128])
    plot.update(3)
    np.testing.assert_allclose([cbar.vmin, cbar.vmax], [-1.7481711049213744, 1.7008913273857975])