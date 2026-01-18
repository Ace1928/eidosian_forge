import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_same_label_error(self):
    overlay = Overlay([Curve(range(10), label='Same').opts(subcoordinate_y=True) for _ in range(2)])
    with pytest.raises(ValueError, match='Elements wrapped in a subcoordinate_y overlay must all have a unique label'):
        bokeh_renderer.get_plot(overlay)