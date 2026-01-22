from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _handle_lookup(self, args, request=None):
    if request is None:
        self._raise_method_deprecation_warning(self.handle_lookup)
    args = list(filter(bool, args))
    lookup = getattr(self, '_lookup', None)
    if args and iscontroller(lookup):
        result = handle_lookup_traversal(lookup, args)
        if result:
            obj, remainder = result
            return lookup_controller(obj, remainder, request)