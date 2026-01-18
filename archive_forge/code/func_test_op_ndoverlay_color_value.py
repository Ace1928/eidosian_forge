import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models import FactorRange, FixedTicker
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.options import AbbreviatedException, Cycle, Palette
from holoviews.element import Curve
from holoviews.plotting.bokeh.callbacks import Callback, PointerXCallback
from holoviews.plotting.util import rgb2hex
from holoviews.streams import PointerX
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_op_ndoverlay_color_value(self):
    colors = ['blue', 'red']
    overlay = NdOverlay({color: Curve(np.arange(i)) for i, color in enumerate(colors)}, 'color').opts('Curve', color='color')
    plot = bokeh_renderer.get_plot(overlay)
    for subplot, color in zip(plot.subplots.values(), colors):
        style = dict(subplot.style[subplot.cyclic_index])
        style = subplot._apply_transforms(subplot.current_frame, {}, {}, style)
        self.assertEqual(style['color'], color)