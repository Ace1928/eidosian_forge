import numpy as np
from bokeh.models import ColorBar
from holoviews.element import Image, QuadMesh
from .test_plot import TestBokehPlot, bokeh_renderer
def test_quadmesh_inverted_coords(self):
    xs = [0, 1, 2]
    ys = [2, 1, 0]
    qmesh = QuadMesh((xs, ys, np.random.rand(3, 3)))
    plot = bokeh_renderer.get_plot(qmesh)
    source = plot.handles['source']
    self.assertEqual(source.data['z'], qmesh.dimension_values(2, flat=False).T.flatten())
    self.assertEqual(source.data['left'], np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5]))
    self.assertEqual(source.data['right'], np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5]))
    self.assertEqual(source.data['top'], np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]))
    self.assertEqual(source.data['bottom'], np.array([-0.5, 0.5, 1.5, -0.5, 0.5, 1.5, -0.5, 0.5, 1.5]))