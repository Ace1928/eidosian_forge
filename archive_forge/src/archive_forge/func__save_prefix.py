import base64
import logging
from io import BytesIO
import bokeh
import param
from bokeh.document import Document
from bokeh.io import curdoc
from bokeh.models import Model
from bokeh.themes.theme import Theme
from panel.io.notebook import render_mimebundle
from panel.io.state import state
from param.parameterized import bothmethod
from ...core import HoloMap, Store
from ..plot import Plot
from ..renderer import HTML_TAGS, MIME_TYPES, Renderer
from .util import compute_plot_size
@bothmethod
def _save_prefix(self_or_cls, ext):
    """Hook to prefix content for instance JS when saving HTML"""
    return