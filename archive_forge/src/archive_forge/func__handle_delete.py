from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _handle_delete(self, method, remainder, request=None):
    """
        Routes ``DELETE`` actions to the appropriate controller.
        """
    if request is None:
        self._raise_method_deprecation_warning(self._handle_delete)
    if remainder:
        match = self._handle_custom_action(method, remainder, request)
        if match:
            return match
        controller = self._lookup_child(remainder[0])
        if controller and (not ismethod(controller)):
            return lookup_controller(controller, remainder[1:], request)
    controller = self._find_controller('post_delete', 'delete')
    if controller:
        return (controller, remainder)
    if remainder:
        if self._find_controller(remainder[0]):
            abort(405)
        sub_controller = self._lookup_child(remainder[0])
        if sub_controller:
            return lookup_controller(sub_controller, remainder[1:], request)
    abort(405)