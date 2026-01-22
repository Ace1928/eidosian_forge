import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
@override
class DBusAnnotationInfo(Gio.DBusAnnotationInfo):
    __init__ = _warn_init(Gio.DBusAnnotationInfo)