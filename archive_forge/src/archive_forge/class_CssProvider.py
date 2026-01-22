import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class CssProvider(Gtk.CssProvider):

    def load_from_data(self, text, length=-1):
        if (Gtk.get_major_version(), Gtk.get_minor_version()) >= (4, 9):
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            super(CssProvider, self).load_from_data(text, length)
        else:
            if isinstance(text, str):
                text = text.encode('utf-8')
            super(CssProvider, self).load_from_data(text)