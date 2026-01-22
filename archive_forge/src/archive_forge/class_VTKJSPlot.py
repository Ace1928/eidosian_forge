from bokeh.core.enums import enumeration
from bokeh.core.has_props import abstract
from bokeh.core.properties import (
from bokeh.models import ColorMapper, Model
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
class VTKJSPlot(AbstractVTKPlot):
    """
    Bokeh model for plotting a 3D scene saved in the `.vtk-js` format
    """
    data = Nullable(Bytes, help='The serialized vtk.js data')
    data_url = Nullable(String, help='The data URL')
    enable_keybindings = Bool(default=False)