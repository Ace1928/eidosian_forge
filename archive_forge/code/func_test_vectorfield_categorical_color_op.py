import numpy as np
from holoviews.element import VectorField
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_vectorfield_categorical_color_op(self):
    vectorfield = VectorField([(0, 0, 0, 1, 'A'), (0, 1, 0, 1, 'B'), (0, 2, 0, 1, 'C')], vdims=['A', 'M', 'color']).opts(color='color')
    plot = bokeh_renderer.get_plot(vectorfield)
    cds = plot.handles['cds']
    glyph = plot.handles['glyph']
    cmapper = plot.handles['color_color_mapper']
    self.assertTrue(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
    self.assertEqual(cds.data['color'], np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'color', 'transform': cmapper})