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
class TestBokehServerSetup(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.current_backend = 'bokeh'
        self.doc = curdoc()
        set_curdoc(Document())
        self.nbcontext = Renderer.notebook_context
        with param.logging_level('ERROR'):
            Renderer.notebook_context = False

    def tearDown(self):
        Store.current_backend = self.previous_backend
        bokeh_renderer.last_plot = None
        Callback._callbacks = {}
        with param.logging_level('ERROR'):
            Renderer.notebook_context = self.nbcontext
        state.curdoc = None
        curdoc().clear()
        set_curdoc(self.doc)
        time.sleep(1)

    def test_render_server_doc_element(self):
        obj = Curve([])
        doc = bokeh_renderer.server_doc(obj)
        self.assertIs(doc, curdoc())
        self.assertIs(bokeh_renderer.last_plot.document, curdoc())

    def test_render_explicit_server_doc_element(self):
        obj = Curve([])
        doc = Document()
        server_doc = bokeh_renderer.server_doc(obj, doc)
        self.assertIs(server_doc, doc)
        self.assertIs(bokeh_renderer.last_plot.document, doc)

    def test_set_up_linked_change_stream_on_server_doc(self):
        obj = Curve([])
        stream = RangeXY(source=obj)
        server_doc = bokeh_renderer.server_doc(obj)
        self.assertIsInstance(server_doc, Document)
        self.assertEqual(len(bokeh_renderer.last_plot.callbacks), 1)
        cb = bokeh_renderer.last_plot.callbacks[0]
        self.assertIsInstance(cb, RangeXYCallback)
        self.assertEqual(cb.streams, [stream])
        assert 'rangesupdate' in bokeh_renderer.last_plot.state._event_callbacks

    def test_set_up_linked_event_stream_on_server_doc(self):
        obj = Curve([])
        stream = PlotReset(source=obj)
        server_doc = bokeh_renderer.server_doc(obj)
        self.assertIsInstance(server_doc, Document)
        cb = bokeh_renderer.last_plot.callbacks[0]
        self.assertIsInstance(cb, ResetCallback)
        self.assertEqual(cb.streams, [stream])