import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class Editable(Gtk.Editable):

    def insert_text(self, text, position):
        return super(Editable, self).insert_text(text, -1, position)
    get_selection_bounds = strip_boolean_result(Gtk.Editable.get_selection_bounds, fail_ret=())