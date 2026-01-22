from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
def __set_parent(self, parent):
    if ismethod(parent):
        self._parent = parent.__self__
    else:
        self._parent = parent