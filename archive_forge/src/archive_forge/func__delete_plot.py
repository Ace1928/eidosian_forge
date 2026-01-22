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
@classmethod
def _delete_plot(cls, plot_id):
    """
        Deletes registered plots and calls Plot.cleanup
        """
    plot = cls._plots.get(plot_id)
    if plot is None:
        return
    plot.cleanup()
    del cls._plots[plot_id]