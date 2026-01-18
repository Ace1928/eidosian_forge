import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_plot_standalone(self):
    standalone = Curve(range(10), label='Data 0').opts(subcoordinate_y=True)
    plot = bokeh_renderer.get_plot(standalone)
    assert (plot.state.x_range.start, plot.state.x_range.end) == (0, 9)
    assert (plot.state.y_range.start, plot.state.y_range.end) == (0, 9)