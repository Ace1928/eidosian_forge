import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreeModelFilter(Gtk.TreeModelFilter):

    def set_visible_func(self, func, data=None):
        super(TreeModelFilter, self).set_visible_func(func, data)

    def set_value(self, iter, column, value):
        iter = self.convert_iter_to_child_iter(iter)
        self.get_model().set_value(iter, column, value)