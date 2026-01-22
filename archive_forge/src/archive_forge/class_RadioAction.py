import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class RadioAction(Gtk.RadioAction):
    __init__ = deprecated_init(Gtk.RadioAction.__init__, arg_names=('name', 'label', 'tooltip', 'stock_id', 'value'), category=PyGTKDeprecationWarning)