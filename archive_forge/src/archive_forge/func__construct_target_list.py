import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def _construct_target_list(targets):
    """Create a list of TargetEntry items from a list of tuples in the form (target, flags, info)

        The list can also contain existing TargetEntry items in which case the existing entry
        is re-used in the return list.
        """
    target_entries = []
    for entry in targets:
        if not isinstance(entry, Gtk.TargetEntry):
            entry = Gtk.TargetEntry.new(*entry)
        target_entries.append(entry)
    return target_entries