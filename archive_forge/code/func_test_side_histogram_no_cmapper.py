import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_side_histogram_no_cmapper(self):
    points = Points(np.random.rand(100, 2))
    plot = bokeh_renderer.get_plot(points.hist())
    plot.initialize_plot()
    adjoint_plot = next(iter(plot.subplots.values()))
    main_plot = adjoint_plot.subplots['main']
    right_plot = adjoint_plot.subplots['right']
    self.assertTrue('color_mapper' not in main_plot.handles)
    self.assertTrue('color_mapper' not in right_plot.handles)