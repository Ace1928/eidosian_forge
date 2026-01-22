from .widget_description import DescriptionStyle, DescriptionWidget
from .valuewidget import ValueWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .trait_types import Color, InstanceDict, TypedTuple
from .utils import deprecation
from traitlets import Unicode, Bool, Int
@register
class HTMLMathStyle(_StringStyle):
    """HTML with math style widget."""
    _model_name = Unicode('HTMLMathStyleModel').tag(sync=True)