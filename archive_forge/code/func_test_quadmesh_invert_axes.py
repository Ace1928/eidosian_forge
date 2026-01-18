import numpy as np
from bokeh.models import ColorBar
from holoviews.element import Image, QuadMesh
from .test_plot import TestBokehPlot, bokeh_renderer
def test_quadmesh_invert_axes(self):
    arr = np.array([[0, 1, 2], [3, 4, 5]])
    qmesh = QuadMesh(Image(arr)).opts(invert_axes=True, tools=['hover'])
    plot = bokeh_renderer.get_plot(qmesh)
    source = plot.handles['source']
    self.assertEqual(source.data['z'], qmesh.dimension_values(2, flat=False).flatten())
    self.assertEqual(source.data['x'], qmesh.dimension_values(0))
    self.assertEqual(source.data['y'], qmesh.dimension_values(1))