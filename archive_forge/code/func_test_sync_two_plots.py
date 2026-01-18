import numpy as np
import pyviz_comms as comms
from bokeh.models import (
from param import concrete_descendents
from holoviews import Curve
from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.plotting.bokeh.element import ElementPlot
from .. import option_intersections
def test_sync_two_plots():
    curve = lambda i: Curve(np.arange(10) * i, label='ABC'[i])
    plot1 = curve(0) * curve(1)
    plot2 = curve(0) * curve(1) * curve(2)
    combined_plot = plot1 + plot2
    grid_bkplot = bokeh_renderer.get_plot(combined_plot).handles['plot']
    for p, *_ in grid_bkplot.children:
        for r in p.renderers:
            if r.name == 'C':
                assert r.js_property_callbacks == {}
            else:
                k, v = next(iter(r.js_property_callbacks.items()))
                assert k == 'change:muted'
                assert len(v) == 1
                assert isinstance(v[0], CustomJS)
                assert v[0].code == 'dst.muted = src.muted'