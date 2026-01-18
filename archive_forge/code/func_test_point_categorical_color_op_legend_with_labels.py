import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_point_categorical_color_op_legend_with_labels(self):
    labels = {'A': 'A point', 'B': 'B point', 'C': 'C point'}
    points = Points([(0, 0, 'A'), (0, 1, 'B'), (0, 2, 'C')], vdims='color').opts(color='color', show_legend=True, legend_labels=labels)
    plot = bokeh_renderer.get_plot(points)
    cds = plot.handles['cds']
    legend = plot.state.legend[0].items[0]
    assert property_to_dict(legend.label) == {'field': '_color_labels'}
    assert cds.data['_color_labels'] == ['A point', 'B point', 'C point']