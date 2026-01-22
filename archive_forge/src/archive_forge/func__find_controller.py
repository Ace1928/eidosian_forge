from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _find_controller(self, *args):
    """
        Returns the appropriate controller for routing a custom action.
        """
    for name in args:
        obj = self._lookup_child(name)
        if obj and iscontroller(obj):
            return obj
    return None