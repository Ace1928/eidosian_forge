from traitlets import Bool, Unicode
from .widget import Widget, widget_serialization, register
from .trait_types import InstanceDict
from .widget_style import Style
from .widget_core import CoreWidget
from .domwidget import DOMWidget
from .utils import deprecation
import warnings
The tooltip information.
        .. deprecated :: 8.0.0
           Use tooltip attribute instead.
        