from .widget_description import DescriptionStyle, DescriptionWidget
from .widget_core import CoreWidget
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .trait_types import Color, InstanceDict
from traitlets import Unicode, Bool, CaselessStrEnum
class _Bool(DescriptionWidget, ValueWidget, CoreWidget):
    """A base class for creating widgets that represent booleans."""
    value = Bool(False, help='Bool value').tag(sync=True)
    disabled = Bool(False, help='Enable or disable user changes.').tag(sync=True)

    def __init__(self, value=None, **kwargs):
        if value is not None:
            kwargs['value'] = value
        super().__init__(**kwargs)
    _model_name = Unicode('BoolModel').tag(sync=True)