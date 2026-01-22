import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class Adjustment(Gtk.Adjustment):
    if GTK3:
        _init = deprecated_init(Gtk.Adjustment.__init__, arg_names=('value', 'lower', 'upper', 'step_increment', 'page_increment', 'page_size'), deprecated_aliases={'page_increment': 'page_incr', 'step_increment': 'step_incr'}, category=PyGTKDeprecationWarning, stacklevel=3)

    def __init__(self, *args, **kwargs):
        if GTK3:
            self._init(*args, **kwargs)
            if 'value' in kwargs:
                self.set_value(kwargs['value'])
            elif len(args) >= 1:
                self.set_value(args[0])
        else:
            Gtk.Adjustment.__init__(self, *args, **kwargs)
            if 'value' in kwargs:
                self.set_value(kwargs['value'])