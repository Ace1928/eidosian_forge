import numpy as np
from bokeh.models import ColorBar
from holoviews.element import Image, QuadMesh
from .test_plot import TestBokehPlot, bokeh_renderer
def test_quadmesh_colormapping(self):
    n = 21
    xs = np.logspace(1, 3, n)
    ys = np.linspace(1, 10, n)
    qmesh = QuadMesh((xs, ys, np.random.rand(n - 1, n - 1)))
    self._test_colormapping(qmesh, 2)