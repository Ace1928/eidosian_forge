from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _lookup_child(self, remainder):
    """
        Lookup a child controller with a named path (handling Unicode paths
        properly for Python 2).
        """
    try:
        controller = getattr(self, remainder, None)
    except UnicodeEncodeError:
        return None
    return controller