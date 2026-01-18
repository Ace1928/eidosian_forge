from .utils import deprecation
from .domwidget import DOMWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import Color, InstanceDict
from traitlets import Unicode, Bool, CaselessStrEnum, Instance, validate, default
Handle a msg from the front-end.

        Parameters
        ----------
        content: dict
            Content of the msg.
        