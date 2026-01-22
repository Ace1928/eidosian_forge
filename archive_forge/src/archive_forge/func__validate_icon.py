from .utils import deprecation
from .domwidget import DOMWidget
from .widget import CallbackDispatcher, register, widget_serialization
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import Color, InstanceDict
from traitlets import Unicode, Bool, CaselessStrEnum, Instance, validate, default
@validate('icon')
def _validate_icon(self, proposal):
    """Strip 'fa-' if necessary'"""
    value = proposal['value']
    if 'fa-' in value:
        deprecation("icons names no longer need 'fa-', just use the class names themselves (for example, 'gear spin' instead of 'fa-gear fa-spin')", internal=['ipywidgets/widgets/', 'traitlets/traitlets.py', '/contextlib.py'])
        value = value.replace('fa-', '')
    return value