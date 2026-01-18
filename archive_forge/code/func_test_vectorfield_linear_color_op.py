import numpy as np
from holoviews.element import VectorField
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vectorfield_linear_color_op(self):
    vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 2, 0, 1, 2)], vdims=['A', 'M', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(vectorfield)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    self.assertTrue(cmapper, LinearColorMapper)
    self.assertEqual(cmapper.low, 0)
    self.assertEqual(cmapper.high, 2)
    self.assertEqual(cds.data['color'], np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})