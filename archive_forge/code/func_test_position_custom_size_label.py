from bokeh.models import LinearAxis, LinearScale, LogAxis, LogScale
from holoviews.element import Curve
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_position_custom_size_label(self):
    overlay = (Curve(range(10), vdims='A').opts(fontsize={'ylabel': '13pt'}) * Curve(range(10), vdims='B').opts(fontsize={'ylabel': '15pt'})).opts(multi_y=True)
    plot = bokeh_renderer.get_plot(overlay)
    assert plot.state.yaxis[0].axis_label == 'A'
    assert plot.state.yaxis[0].axis_label_text_font_size == '13pt'
    assert plot.state.yaxis[1].axis_label == 'B'
    assert plot.state.yaxis[1].axis_label_text_font_size == '15pt'