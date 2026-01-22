from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _raise_method_deprecation_warning(self, handler):
    warnings.warn('The function signature for %s.%s.%s is changing in the next version of pecan.\nPlease update to: `%s(self, method, remainder, request)`.' % (self.__class__.__module__, self.__class__.__name__, handler.__name__, handler.__name__), DeprecationWarning)