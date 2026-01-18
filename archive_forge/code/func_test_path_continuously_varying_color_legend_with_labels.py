import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import HoloMap, NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Contours, Path, Polygons
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_path_continuously_varying_color_legend_with_labels(self):
    data = {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'cat': [0, 1, 2, 0, 1, 2, 0, 1, 2]}
    colors = ['#FF0000', '#00FF00', '#0000FF']
    levels = [0, 1, 2, 3]
    path = Path(data, vdims='cat').opts(color='cat', cmap=dict(zip(levels, colors)), line_width=4, show_legend=True, legend_labels={0: 'A', 1: 'B', 2: 'C'})
    plot = bokeh_renderer.get_plot(path)
    cds = plot.handles['cds']
    item = plot.state.legend[0].items[0]
    legend = {'field': '_color_str___labels'}
    self.assertEqual(cds.data['_color_str___labels'], ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'])
    self.assertEqual(property_to_dict(item.label), legend)
    self.assertEqual(item.renderers, [plot.handles['glyph_renderer']])