from .widget_description import DescriptionStyle, DescriptionWidget
from .valuewidget import ValueWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .trait_types import Color, InstanceDict, TypedTuple
from .utils import deprecation
from traitlets import Unicode, Bool, Int
class _StringStyle(DescriptionStyle, CoreWidget):
    """Text input style widget."""
    _model_name = Unicode('StringStyleModel').tag(sync=True)
    background = Unicode(None, allow_none=True, help='Background specifications.').tag(sync=True)
    font_size = Unicode(None, allow_none=True, help='Text font size.').tag(sync=True)
    text_color = Color(None, allow_none=True, help='Text color').tag(sync=True)