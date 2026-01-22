from .widget_description import DescriptionStyle, DescriptionWidget
from .widget_core import CoreWidget
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .trait_types import Color, InstanceDict
from traitlets import Unicode, Bool, CaselessStrEnum
@register
class CheckboxStyle(DescriptionStyle, CoreWidget):
    """Checkbox widget style."""
    _model_name = Unicode('CheckboxStyleModel').tag(sync=True)
    background = Unicode(None, allow_none=True, help='Background specifications.').tag(sync=True)