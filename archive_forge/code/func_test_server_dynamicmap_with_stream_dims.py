import asyncio
import time
import param
import pytest
from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.io.doc import curdoc, set_curdoc
from bokeh.models import ColumnDataSource
from panel import serve
from panel.io.state import state
from panel.widgets import DiscreteSlider, FloatSlider
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HLine, Path, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.plotting.bokeh.callbacks import Callback, RangeXYCallback, ResetCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import PlotReset, RangeXY, Stream
def test_server_dynamicmap_with_stream_dims(self):
    stream = Stream.define('Custom', y=2)()
    dmap = DynamicMap(lambda x, y: Curve([x, 1, y]), kdims=['x', 'y'], streams=[stream]).redim.values(x=[1, 2, 3])
    obj, _ = bokeh_renderer._validate(dmap, None)
    _, session = self._launcher(obj, port=6006)
    [(doc, _)] = obj._documents.items()
    orig_cds = session.document.roots[0].select_one({'type': ColumnDataSource})
    self.assertEqual(orig_cds.data['y'][2], 2)

    def run():
        stream.event(y=3)
    doc.add_next_tick_callback(run)
    time.sleep(1)
    cds = self.session.document.roots[0].select_one({'type': ColumnDataSource})
    self.assertEqual(cds.data['y'][2], 3)
    self.assertEqual(orig_cds.data['y'][0], 1)
    slider = obj.layout.select(DiscreteSlider)[0]

    def run():
        slider.value = 3
    doc.add_next_tick_callback(run)
    time.sleep(1)
    cds = self.session.document.roots[0].select_one({'type': ColumnDataSource})
    self.assertEqual(cds.data['y'][0], 3)