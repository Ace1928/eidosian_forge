import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, DatetimeAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay
from holoviews.element import Dataset, Histogram, Image, Points
from holoviews.operation import histogram
from holoviews.plotting.bokeh.util import property_to_dict
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_side_histogram_cmapper_weighted(self):
    """Assert weighted histograms share colormapper"""
    x, y = np.mgrid[-50:51, -50:51] * 0.1
    img = Image(np.sin(x ** 2 + y ** 2), bounds=(-1, -1, 1, 1))
    adjoint = img.hist(dimension=['x', 'y'], weight_dimension='z', mean_weighted=True)
    plot = bokeh_renderer.get_plot(adjoint)
    plot.initialize_plot()
    adjoint_plot = next(iter(plot.subplots.values()))
    main_plot = adjoint_plot.subplots['main']
    right_plot = adjoint_plot.subplots['right']
    top_plot = adjoint_plot.subplots['top']
    self.assertIs(main_plot.handles['color_mapper'], right_plot.handles['color_mapper'])
    self.assertIs(main_plot.handles['color_mapper'], top_plot.handles['color_mapper'])
    self.assertEqual(main_plot.handles['color_dim'], img.vdims[0])