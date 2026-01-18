import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_overlaid_without_label_no_error(self):
    overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
    with_span = overlay * VSpan(1, 2)
    bokeh_renderer.get_plot(with_span)