from bokeh.core.enums import enumeration
from bokeh.core.has_props import abstract
from bokeh.core.properties import (
from bokeh.models import ColorMapper, Model
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
class VTKSynchronizedPlot(AbstractVTKPlot):
    """
    Bokeh model for plotting a VTK render window
    """
    arrays = Dict(String, Bytes)
    arrays_processed = List(String)
    enable_keybindings = Bool(default=False)
    one_time_reset = Bool(default=False)
    rebuild = Bool(default=False, help='If true when scene change all the render is rebuilt from scratch')
    scene = Dict(String, Any, help='The serialized vtk.js scene on json format')