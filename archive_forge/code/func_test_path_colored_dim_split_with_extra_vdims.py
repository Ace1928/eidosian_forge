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
def test_path_colored_dim_split_with_extra_vdims(self):
    xs = [1, 2, 3, 4]
    ys = xs[::-1]
    color = [0, 0.25, 0.5, 0.75]
    other = ['A', 'B', 'C', 'D']
    data = {'x': xs, 'y': ys, 'color': color, 'other': other}
    path = Path([data], vdims=['color', 'other']).opts(color=dim('color') * 2, tools=['hover'])
    plot = bokeh_renderer.get_plot(path)
    source = plot.handles['source']
    self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
    self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
    self.assertEqual(source.data['other'], np.array(['A', 'B', 'C']))
    self.assertEqual(source.data['color'], np.array([0, 0.5, 1]))