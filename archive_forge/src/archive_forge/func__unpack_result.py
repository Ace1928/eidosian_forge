import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
@classmethod
def _unpack_result(klass, result):
    """Convert a D-BUS return variant into an appropriate return value"""
    result = result.unpack()
    if len(result) == 1:
        result = result[0]
    elif len(result) == 0:
        result = None
    return result