from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _handle_post(self, method, remainder, request=None):
    """
        Routes ``POST`` requests.
        """
    if request is None:
        self._raise_method_deprecation_warning(self._handle_post)
    if remainder:
        match = self._handle_custom_action(method, remainder, request)
        if match:
            return match
        controller = self._lookup_child(remainder[0])
        if controller and (not ismethod(controller)):
            return lookup_controller(controller, remainder[1:], request)
    controller = self._find_controller(method)
    if controller:
        return (controller, remainder)
    abort(405)