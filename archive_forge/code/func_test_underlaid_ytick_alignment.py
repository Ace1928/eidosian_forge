import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_underlaid_ytick_alignment(self):
    overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
    with_span = VSpan(1, 2) * overlay
    plot = bokeh_renderer.get_plot(with_span)
    assert plot.state.yaxis.ticker.ticks == [0, 1]