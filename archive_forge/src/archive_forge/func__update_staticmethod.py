from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def _update_staticmethod(self, oldsm, newsm):
    """Update a staticmethod update."""
    self._update(None, None, oldsm.__get__(0), newsm.__get__(0))