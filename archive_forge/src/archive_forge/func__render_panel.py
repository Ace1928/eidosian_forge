import base64
import os
from contextlib import contextmanager
from functools import partial
from io import BytesIO, StringIO
import panel as pn
import param
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.io import curdoc
from bokeh.resources import CDN, INLINE
from packaging.version import Version
from panel import config
from panel.io.notebook import ipywidget, load_notebook, render_mimebundle, render_model
from panel.io.state import state
from panel.models.comm_manager import CommManager as PnCommManager
from panel.pane import HoloViews as HoloViewsPane
from panel.viewable import Viewable
from panel.widgets.player import PlayerBase
from pyviz_comms import CommManager
from param.parameterized import bothmethod
from ..core import AdjointLayout, DynamicMap, HoloMap, Layout
from ..core.data import disable_pipeline
from ..core.io import Exporter
from ..core.options import Compositor, SkipRendering, Store, StoreOptions
from ..core.util import unbound_dimensions
from ..streams import Stream
from . import Plot
from .util import collate, displayable, initialize_dynamic
def _render_panel(self, plot, embed=False, comm=True):
    comm = self.comm_manager.get_server_comm() if comm else None
    doc = Document()
    with config.set(embed=embed):
        model = plot.layout._render_model(doc, comm)
    if embed:
        return render_model(model, comm)
    ref = model.ref['id']
    manager = PnCommManager(comm_id=comm.id, plot_id=ref)
    client_comm = self.comm_manager.get_client_comm(on_msg=partial(plot._on_msg, ref, manager), on_error=partial(plot._on_error, ref), on_stdout=partial(plot._on_stdout, ref), on_open=lambda _: comm.init())
    manager.client_comm_id = client_comm.id
    return render_mimebundle(model, doc, comm, manager)