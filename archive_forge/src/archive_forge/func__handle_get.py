from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _handle_get(self, method, remainder, request=None):
    """
        Routes ``GET`` actions to the appropriate controller.
        """
    if request is None:
        self._raise_method_deprecation_warning(self._handle_get)
    if not remainder or remainder == ['']:
        remainder = list(filter(bool, remainder))
        controller = self._find_controller('get_all', 'get')
        if controller:
            self._handle_bad_rest_arguments(controller, remainder, request)
            return (controller, [])
        abort(405)
    method_name = remainder[-1]
    if method_name in ('new', 'edit', 'delete'):
        if method_name == 'delete':
            method_name = 'get_delete'
        controller = self._find_controller(method_name)
        if controller:
            return (controller, remainder[:-1])
    match = self._handle_custom_action(method, remainder, request)
    if match:
        return match
    controller = self._lookup_child(remainder[0])
    if controller and (not ismethod(controller)):
        return lookup_controller(controller, remainder[1:], request)
    controller = self._find_controller('get_one', 'get')
    if controller:
        self._handle_bad_rest_arguments(controller, remainder, request)
        return (controller, remainder)
    abort(405)