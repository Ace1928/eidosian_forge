from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _find_sub_controllers(self, remainder, request):
    """
        Identifies the correct controller to route to by analyzing the
        request URI.
        """
    method = None
    for name in ('get_one', 'get'):
        if hasattr(self, name):
            method = name
            break
    if not method:
        return
    args = self._get_args_for_controller(getattr(self, method))
    fixed_args = len(args) - len(request.pecan.get('routing_args', []))
    var_args = getargspec(getattr(self, method)).varargs
    if var_args:
        for i, item in enumerate(remainder):
            controller = self._lookup_child(item)
            if controller and (not ismethod(controller)):
                self._set_routing_args(request, remainder[:i])
                return lookup_controller(controller, remainder[i + 1:], request)
    elif fixed_args < len(remainder) and hasattr(self, remainder[fixed_args]):
        controller = self._lookup_child(remainder[fixed_args])
        if not ismethod(controller):
            self._set_routing_args(request, remainder[:fixed_args])
            return lookup_controller(controller, remainder[fixed_args + 1:], request)