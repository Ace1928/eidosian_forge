import numpy as np
from holoviews.element import VectorField
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vectorfield_alpha_op(self):
    vectorfield = VectorField([(0, 0, 0, 1, 0), (0, 1, 0, 1, 0.2), (0, 2, 0, 1, 0.7)], vdims=['A', 'M', 'alpha']).opts(alpha='alpha')
    plot = bokeh_renderer.get_plot(vectorfield)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    self.assertEqual(cds.data['alpha'], np.array([0, 0.2, 0.7, 0, 0.2, 0.7, 0, 0.2, 0.7]))
    self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'alpha'})