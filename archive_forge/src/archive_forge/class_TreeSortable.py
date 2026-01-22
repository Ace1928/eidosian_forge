import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
class TreeSortable(Gtk.TreeSortable):
    get_sort_column_id = strip_boolean_result(Gtk.TreeSortable.get_sort_column_id, fail_ret=(None, None))

    def set_sort_func(self, sort_column_id, sort_func, user_data=None):
        super(TreeSortable, self).set_sort_func(sort_column_id, sort_func, user_data)

    def set_default_sort_func(self, sort_func, user_data=None):
        super(TreeSortable, self).set_default_sort_func(sort_func, user_data)