from .widget_description import DescriptionStyle, DescriptionWidget
from .valuewidget import ValueWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .trait_types import Color, InstanceDict, TypedTuple
from .utils import deprecation
from traitlets import Unicode, Bool, Int
class _String(DescriptionWidget, ValueWidget, CoreWidget):
    """Base class used to create widgets that represent a string."""
    value = Unicode(help='String value').tag(sync=True)
    placeholder = Unicode('\u200b', help='Placeholder text to display when nothing has been typed').tag(sync=True)
    style = InstanceDict(_StringStyle).tag(sync=True, **widget_serialization)

    def __init__(self, value=None, **kwargs):
        if value is not None:
            kwargs['value'] = value
        super().__init__(**kwargs)
    _model_name = Unicode('StringModel').tag(sync=True)