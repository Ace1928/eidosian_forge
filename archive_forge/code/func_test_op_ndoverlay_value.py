import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_op_ndoverlay_value(self):
    colors = ['blue', 'red']
    overlay = NdOverlay({color: Bars(np.arange(i + 2)) for i, color in enumerate(colors)}, 'Color').opts('Bars', fill_color='Color')
    plot = bokeh_renderer.get_plot(overlay)
    for subplot, color in zip(plot.subplots.values(), colors):
        self.assertEqual(subplot.handles['glyph'].fill_color, color)