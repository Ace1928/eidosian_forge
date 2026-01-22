from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _handle_unknown_method(self, method, remainder, request=None):
    """
        Routes undefined actions (like TRACE) to the appropriate controller.
        """
    if request is None:
        self._raise_method_deprecation_warning(self._handle_unknown_method)
    controller = self._find_controller('post_%s' % method, method)
    if controller:
        return (controller, remainder)
    if remainder:
        if self._find_controller(remainder[0]):
            abort(405)
        sub_controller = self._lookup_child(remainder[0])
        if sub_controller:
            return lookup_controller(sub_controller, remainder[1:], request)
    abort(405)