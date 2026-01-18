from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller

        Returns `True` or `False` to grant access.  Implemented in subclasses
        of :class:`SecureController`.
        