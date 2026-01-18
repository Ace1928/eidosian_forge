import numpy as np
from holoviews.element import VectorField
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vectorfield_line_width_op(self):
    vectorfield = VectorField([(0, 0, 0, 1, 1), (0, 1, 0, 1, 4), (0, 2, 0, 1, 8)], vdims=['A', 'M', 'line_width']).opts(line_width='line_width')
    plot = bokeh_renderer.get_plot(vectorfield)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['line_width'], np.array([1, 4, 8, 1, 4, 8, 1, 4, 8]))
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})