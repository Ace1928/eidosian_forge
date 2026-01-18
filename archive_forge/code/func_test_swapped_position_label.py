from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_swapped_position_label(self):
    overlay = (Curve(range(10), vdims=['A']).opts(yaxis='right') * Curve(range(10), vdims=['B']).opts(yaxis='left')).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    assert plot.state.yaxis[0].axis_label == 'B'
    assert plot.state.yaxis[1].axis_label == 'A'