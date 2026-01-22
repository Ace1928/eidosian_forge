from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Instance
from .trait_types import Color, InstanceDict, NumberFormat
from traitlets import (
@register
class SliderStyle(DescriptionStyle, CoreWidget):
    """Button style widget."""
    _model_name = Unicode('SliderStyleModel').tag(sync=True)
    handle_color = Color(None, allow_none=True, help='Color of the slider handle.').tag(sync=True)