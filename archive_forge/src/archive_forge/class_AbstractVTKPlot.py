from bokeh.core.enums import enumeration
from bokeh.core.has_props import abstract
from bokeh.core.properties import (
from bokeh.models import ColorMapper, Model
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
@abstract
class AbstractVTKPlot(HTMLBox):
    """
    Abstract Bokeh model for vtk plots that wraps around a vtk-js library and
    renders it inside a Bokeh plot.
    """
    __javascript_raw__ = [vtk_cdn]

    @classproperty
    def __javascript__(cls):
        return bundled_files(AbstractVTKPlot)

    @classproperty
    def __js_skip__(cls):
        return {'vtk': cls.__javascript__}
    __js_require__ = {'paths': {'vtk': vtk_cdn[:-3]}, 'shim': {'vtk': {'exports': 'vtk'}}}
    axes = Instance(VTKAxes)
    camera = Dict(String, Any)
    color_mappers = List(Instance(ColorMapper))
    height = Override(default=300)
    orientation_widget = Bool(default=False)
    interactive_orientation_widget = Bool(default=False)
    width = Override(default=300)
    annotations = List(Dict(String, Any))